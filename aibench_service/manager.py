from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .executor import DockerJobExecutor, ExecutionContext, ExecutionError, JobExecutor
from .schemas import JobRequest
from .store import JobNotFoundError, JobStore


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timed_out"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(
        self,
        store: JobStore,
        executor: Optional[JobExecutor] = None,
        max_workers: int = 2,
    ):
        self.store = store
        self.executor = executor or DockerJobExecutor()
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aibench-job")
        self._cancel_events: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._recover_interrupted_jobs()

    def submit(self, request: JobRequest) -> Dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex}"
        now = utc_now()
        state = {
            "job_id": job_id,
            "external_task_id": request.external_task_id,
            "status": "queued",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "metrics": {},
            "error": None,
            "container": {},
            "artifacts": self._artifacts(job_id),
        }
        self.store.create(job_id, request.model_dump(mode="json"), state)
        event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = event
        self.pool.submit(self._run, job_id, request, event)
        return state

    def get(self, job_id: str) -> Dict[str, Any]:
        return self.store.read_state(job_id)

    def result(self, job_id: str) -> Dict[str, Any]:
        state = self.get(job_id)
        if state["status"] != "completed":
            raise RuntimeError(f"result is not available while job status is {state['status']}")
        return self.store.read_result(job_id)

    def logs(self, job_id: str) -> str:
        self.get(job_id)
        return self.store.read_logs(job_id)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        state = self.get(job_id)
        if state["status"] in TERMINAL_STATUSES:
            return state
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event:
            event.set()
        self.executor.cancel(job_id)
        return self._update(job_id, status="cancelled", progress=state.get("progress", 0))

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, request: JobRequest, event: threading.Event) -> None:
        try:
            if event.is_set():
                return
            self._update(job_id, status="preparing", progress=10)
            context = ExecutionContext(
                job_id=job_id,
                job_dir=self.store.job_dir(job_id),
                cancel_event=event,
                update=lambda **values: self._update(job_id, **values),
                append_log=lambda text: self.store.append_log(job_id, text),
            )
            result = self.executor.execute(request, context)
            if event.is_set():
                self._update(job_id, status="cancelled")
                return
            metrics = result.get("metrics") or result.get("evaluation_results", {}).get("metrics") or {}
            container = result.get("container") or self.get(job_id).get("container") or {}
            self.store.write_result(job_id, result)
            self._update(
                job_id,
                status="completed",
                progress=100,
                metrics=metrics,
                container=container,
                error=None,
            )
        except ExecutionError as exc:
            self.store.append_log(job_id, f"ERROR [{exc.stage}] {exc}")
            current = self.get(job_id)
            status = "cancelled" if event.is_set() else exc.status
            if current.get("status") == "cancelled":
                status = "cancelled"
            self._update(
                job_id,
                status=status,
                error={"stage": exc.stage, "message": str(exc)},
            )
        except Exception as exc:
            self.store.append_log(job_id, f"ERROR [internal] {exc}")
            current = self.get(job_id)
            if current.get("status") != "cancelled":
                self._update(
                    job_id,
                    status="failed",
                    error={"stage": "internal", "message": str(exc)},
                )
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)

    def _update(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        updates["updated_at"] = utc_now()
        return self.store.update_state(job_id, **updates)

    def _recover_interrupted_jobs(self) -> None:
        for job_id in self.store.list_job_ids():
            try:
                state = self.store.read_state(job_id)
            except (JobNotFoundError, ValueError):
                continue
            if state.get("status") not in TERMINAL_STATUSES:
                self._update(
                    job_id,
                    status="failed",
                    error={
                        "stage": "agent_restart",
                        "message": "AIBenchAgent restarted before the job completed",
                    },
                )

    def _artifacts(self, job_id: str) -> Dict[str, str]:
        directory = self.store.job_dir(job_id)
        return {
            "request": str(directory / "request.json"),
            "state": str(directory / "state.json"),
            "result": str(directory / "result.json"),
            "container_log": str(directory / "container.log"),
        }
