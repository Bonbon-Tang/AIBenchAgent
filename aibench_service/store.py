from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable


class JobNotFoundError(KeyError):
    pass


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in job_id):
            raise ValueError("invalid job id")
        return self.root / job_id

    def create(self, job_id: str, request: Dict[str, Any], state: Dict[str, Any]) -> None:
        with self._lock:
            directory = self.job_dir(job_id)
            directory.mkdir(parents=False, exist_ok=False)
            self._atomic_write(directory / "request.json", request)
            self._atomic_write(directory / "state.json", state)
            (directory / "container.log").touch()

    def read_request(self, job_id: str) -> Dict[str, Any]:
        return self._read_json(self.job_dir(job_id) / "request.json", job_id)

    def read_state(self, job_id: str) -> Dict[str, Any]:
        return self._read_json(self.job_dir(job_id) / "state.json", job_id)

    def update_state(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            state = self.read_state(job_id)
            state.update(updates)
            self._atomic_write(self.job_dir(job_id) / "state.json", state)
            return state

    def write_result(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(self.job_dir(job_id) / "result.json", result)

    def read_result(self, job_id: str) -> Dict[str, Any]:
        return self._read_json(self.job_dir(job_id) / "result.json", job_id)

    def append_log(self, job_id: str, text: str) -> None:
        with self._lock:
            with (self.job_dir(job_id) / "container.log").open("a", encoding="utf-8") as handle:
                handle.write(text)
                if text and not text.endswith("\n"):
                    handle.write("\n")

    def read_logs(self, job_id: str, max_bytes: int = 1_000_000) -> str:
        path = self.job_dir(job_id) / "container.log"
        if not path.exists():
            raise JobNotFoundError(job_id)
        with path.open("rb") as handle:
            size = path.stat().st_size
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return handle.read().decode("utf-8", errors="replace")

    def list_job_ids(self) -> Iterable[str]:
        return [path.name for path in self.root.iterdir() if path.is_dir()]

    def _read_json(self, path: Path, job_id: str) -> Dict[str, Any]:
        if not path.exists():
            raise JobNotFoundError(job_id)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return value

    @staticmethod
    def _atomic_write(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
