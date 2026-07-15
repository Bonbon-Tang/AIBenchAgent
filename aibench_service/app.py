from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status

from .executor import JobExecutor
from .manager import JobManager
from .schemas import JobRequest, JobView, LogsView
from .store import JobNotFoundError, JobStore


def create_app(
    *,
    data_dir: Optional[str | Path] = None,
    executor: Optional[JobExecutor] = None,
    max_workers: Optional[int] = None,
) -> FastAPI:
    root = Path(data_dir or os.getenv("AIBENCH_DATA_DIR", "./var/jobs"))
    workers = max_workers or int(os.getenv("AIBENCH_MAX_WORKERS", "2"))
    manager = JobManager(JobStore(root), executor=executor, max_workers=workers)

    app = FastAPI(title="AIBenchAgent Execution Service", version="1.0.0")
    app.state.job_manager = manager

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/jobs", response_model=JobView, status_code=status.HTTP_202_ACCEPTED)
    def submit_job(request: JobRequest):
        return manager.submit(request)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobView)
    def get_job(job_id: str):
        try:
            return manager.get(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.get("/api/v1/jobs/{job_id}/result")
    def get_result(job_id: str):
        try:
            return manager.result(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/v1/jobs/{job_id}/logs", response_model=LogsView)
    def get_logs(job_id: str):
        try:
            return {"job_id": job_id, "logs": manager.logs(job_id)}
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobView)
    def cancel_job(job_id: str):
        try:
            return manager.cancel(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.on_event("shutdown")
    def shutdown():
        manager.shutdown()

    return app
