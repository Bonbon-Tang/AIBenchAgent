import json
import threading
import time

from fastapi.testclient import TestClient

from aibench_service.app import create_app
from aibench_service.executor import DockerJobExecutor


def payload():
    return {
        "external_task_id": "projectten-42",
        "task_type": "model_deployment",
        "scenario": "llm",
        "chip": "Ascend_910B",
        "chip_count": 1,
        "image": {
            "name": "example/ascend-model:test",
            "command": ["python3", "/workspace/benchmark.py"],
        },
        "resources": {"device_ids": [0], "gpus": None, "timeout_seconds": 30},
    }


class SuccessfulExecutor:
    def execute(self, request, context):
        context.update(status="running", progress=50, container={"name": "fake-container"})
        context.append_log("benchmark started")
        return {
            "status": "completed",
            "metrics": {"throughput": 12.5, "avg_latency_ms": 80.0},
            "container": {"name": "fake-container"},
        }

    def cancel(self, job_id):
        return None


class BlockingExecutor:
    def __init__(self):
        self.started = threading.Event()

    def execute(self, request, context):
        self.started.set()
        while not context.cancel_event.wait(0.01):
            pass
        return {"status": "completed", "metrics": {}}

    def cancel(self, job_id):
        return None


def wait_for(client, job_id, expected, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        if response.json()["status"] in expected:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_submit_execute_and_read_persisted_result(tmp_path):
    app = create_app(data_dir=tmp_path, executor=SuccessfulExecutor(), max_workers=1)
    with TestClient(app) as client:
        response = client.post("/api/v1/jobs", json=payload())
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        state = wait_for(client, job_id, {"completed"})
        assert state["progress"] == 100
        assert state["metrics"]["throughput"] == 12.5
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.json()["metrics"]["avg_latency_ms"] == 80.0
        logs = client.get(f"/api/v1/jobs/{job_id}/logs").json()["logs"]
        assert "benchmark started" in logs
        assert json.loads((tmp_path / job_id / "request.json").read_text())["external_task_id"] == "projectten-42"
        assert (tmp_path / job_id / "state.json").exists()
        assert (tmp_path / job_id / "result.json").exists()
        assert (tmp_path / job_id / "container.log").exists()


def test_cancel_running_job(tmp_path):
    executor = BlockingExecutor()
    app = create_app(data_dir=tmp_path, executor=executor, max_workers=1)
    with TestClient(app) as client:
        job_id = client.post("/api/v1/jobs", json=payload()).json()["job_id"]
        assert executor.started.wait(1)
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        state = wait_for(client, job_id, {"cancelled"})
        assert state["status"] == "cancelled"
        assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 409


def test_docker_command_uses_ascend_devices_and_no_gpu_flag(tmp_path):
    request_payload = payload()
    from aibench_service.schemas import JobRequest

    request = JobRequest.model_validate(request_payload)
    executor = DockerJobExecutor()
    command = executor.build_command(
        request,
        tmp_path,
        "aibench-job-test",
        executor._resolve_devices(request),
    )
    assert "--gpus" not in command
    assert "/dev/davinci0" in command
    assert "/dev/davinci_manager" in command
    assert f"{tmp_path.resolve()}:/workspace/results" in command
    assert command[-2:] == ["python3", "/workspace/benchmark.py"]


def test_rejects_non_ascend_device_mapping(tmp_path):
    request_payload = payload()
    request_payload["resources"]["devices"] = ["/dev/sda"]
    from aibench_service.executor import ExecutionError
    from aibench_service.schemas import JobRequest

    executor = DockerJobExecutor()
    request = JobRequest.model_validate(request_payload)
    try:
        executor._resolve_devices(request)
    except ExecutionError as exc:
        assert exc.stage == "preflight"
    else:
        raise AssertionError("unsafe device mapping was accepted")


def test_rejects_volume_outside_allowlist(tmp_path):
    request_payload = payload()
    request_payload["image"]["volumes"] = ["/:/host:ro"]
    from aibench_service.executor import ExecutionError
    from aibench_service.schemas import JobRequest

    executor = DockerJobExecutor(allowed_volume_roots=["/models"])
    request = JobRequest.model_validate(request_payload)
    try:
        executor.build_command(
            request,
            tmp_path,
            "aibench-job-test",
            executor._resolve_devices(request),
        )
    except ExecutionError as exc:
        assert exc.stage == "preflight"
    else:
        raise AssertionError("unsafe volume mapping was accepted")
