from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from .schemas import JobRequest


class ExecutionError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "execute", status: str = "failed"):
        super().__init__(message)
        self.stage = stage
        self.status = status


@dataclass
class ExecutionContext:
    job_id: str
    job_dir: Path
    cancel_event: threading.Event
    update: Callable[..., None]
    append_log: Callable[[str], None]


class JobExecutor(Protocol):
    def execute(self, request: JobRequest, context: ExecutionContext) -> Dict[str, Any]: ...

    def cancel(self, job_id: str) -> None: ...


class DockerJobExecutor:
    _DEVICE_PATTERN = re.compile(
        r"^/dev/(davinci(?:[0-9]+|_manager)|devmm_svm|hisi_hdc)$"
    )

    def __init__(self, docker_binary: str = "docker", allowed_volume_roots: List[str] | None = None):
        self.docker_binary = docker_binary
        configured_roots = allowed_volume_roots or os.getenv(
            "AIBENCH_ALLOWED_VOLUME_ROOTS",
            "/models,/opt/aibench/workloads,/usr/local/Ascend",
        ).split(",")
        self.allowed_volume_roots = [Path(root).resolve() for root in configured_roots if root]
        self._containers: Dict[str, str] = {}
        self._lock = threading.Lock()

    def execute(self, request: JobRequest, context: ExecutionContext) -> Dict[str, Any]:
        container_name = self._container_name(context.job_id)
        devices = self._resolve_devices(request)
        command = self.build_command(request, context.job_dir, container_name, devices)
        context.update(status="starting_container", progress=20, container={"name": container_name})
        context.append_log("docker command: " + " ".join(command))

        with self._lock:
            self._containers[context.job_id] = container_name

        started = time.monotonic()
        timed_out = False
        process = None
        try:
            with (context.job_dir / "container.log").open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                context.update(status="running", progress=45)
                timeout = request.resources.timeout_seconds
                while process.poll() is None:
                    if context.cancel_event.wait(0.25):
                        self._remove_container(container_name)
                        raise ExecutionError("job cancelled", stage="execute", status="cancelled")
                    if time.monotonic() - started > timeout:
                        timed_out = True
                        self._remove_container(container_name)
                        raise ExecutionError(
                            f"container exceeded timeout of {timeout} seconds",
                            stage="execute",
                            status="timed_out",
                        )
                if process.returncode != 0:
                    raise ExecutionError(
                        f"container exited with code {process.returncode}",
                        stage="container",
                    )

            context.update(status="collecting", progress=85)
            result_path = context.job_dir / "result.json"
            if not result_path.exists():
                raise ExecutionError(
                    "container did not write /workspace/results/result.json",
                    stage="collect_results",
                )
            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ExecutionError(f"invalid result.json: {exc}", stage="collect_results") from exc
            if not isinstance(result, dict):
                raise ExecutionError("result.json must contain a JSON object", stage="collect_results")
            if str(result.get("status", "completed")).lower() in {"failed", "error"}:
                raise ExecutionError(
                    str(result.get("error") or "container reported a failed result"),
                    stage="benchmark",
                )
            metrics = result.get("metrics")
            if not isinstance(metrics, dict):
                raise ExecutionError(
                    "result.json must contain a metrics object",
                    stage="collect_results",
                )
            result.setdefault("container", {})
            result["container"].update({"name": container_name})
            return result
        except FileNotFoundError as exc:
            raise ExecutionError(
                f"docker executable not found: {self.docker_binary}",
                stage="preflight",
            ) from exc
        finally:
            with self._lock:
                self._containers.pop(context.job_id, None)
            if request.resources.cleanup_container and not timed_out:
                self._remove_container(container_name)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            container_name = self._containers.get(job_id)
        if container_name:
            self._remove_container(container_name)

    def build_command(
        self,
        request: JobRequest,
        job_dir: Path,
        container_name: str,
        devices: List[str],
    ) -> List[str]:
        resources = request.resources
        image = request.image
        command = [self.docker_binary, "run", "--name", container_name]
        if resources.network_mode:
            command.extend(["--network", resources.network_mode])
        if resources.ipc_mode:
            command.append(f"--ipc={resources.ipc_mode}")
        if resources.gpus:
            command.extend(["--gpus", resources.gpus])
        if resources.privileged:
            command.append("--privileged=true")
        if resources.shm_size:
            command.extend(["--shm-size", resources.shm_size])
        for device in devices:
            command.extend(["--device", device])
        for extra_dev in getattr(resources, "extra_devices", []):
            command.extend(["--device", extra_dev])
        command.extend(["-v", f"{job_dir.resolve()}:/workspace/results"])
        for volume in image.volumes:
            self._validate_volume(volume)
            command.extend(["-v", volume])
        for extra_vol in getattr(resources, "extra_volumes", []):
            self._validate_volume(extra_vol)
            command.extend(["-v", extra_vol])
        for key, value in image.environment.items():
            command.extend(["-e", f"{key}={value}"])
        if image.working_dir:
            command.extend(["-w", image.working_dir])
        if image.entrypoint:
            command.extend(["--entrypoint", image.entrypoint])
        command.append(image.name)
        if isinstance(image.command, str):
            command.extend(["/bin/bash", "-lc", image.command])
        else:
            command.extend(image.command)
        return command

    def _resolve_devices(self, request: JobRequest) -> List[str]:
        devices = list(request.resources.devices)
        if not devices:
            devices = [f"/dev/davinci{device_id}" for device_id in request.resources.device_ids]
            devices.extend(["/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"])
        invalid = [device for device in devices if not self._DEVICE_PATTERN.fullmatch(device)]
        if invalid:
            raise ExecutionError(
                f"unsupported device mapping: {', '.join(invalid)}",
                stage="preflight",
            )
        return list(dict.fromkeys(devices))

    def _validate_volume(self, volume: str) -> None:
        host_path = volume.split(":", 1)[0]
        if not host_path.startswith("/"):
            raise ExecutionError(f"volume host path must be absolute: {volume}", stage="preflight")
        resolved = Path(host_path).resolve()
        if not any(resolved == root or root in resolved.parents for root in self.allowed_volume_roots):
            allowed = ", ".join(str(root) for root in self.allowed_volume_roots)
            raise ExecutionError(
                f"volume host path is not allowed: {host_path}; allowed roots: {allowed}",
                stage="preflight",
            )

    def _remove_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                [self.docker_binary, "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _container_name(job_id: str) -> str:
        return f"aibench-{job_id}"[:63]
