from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImageSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    command: Union[str, List[str]]
    entrypoint: Optional[str] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    volumes: List[str] = Field(default_factory=list)
    working_dir: Optional[str] = None


class ResourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_ids: List[int] = Field(default_factory=lambda: [0])
    devices: List[str] = Field(default_factory=list)
    gpus: Optional[str] = None
    network_mode: Optional[str] = "host"
    ipc_mode: Optional[str] = "host"
    timeout_seconds: int = Field(default=1800, ge=1, le=86400)
    cleanup_container: bool = True

    @field_validator("device_ids")
    @classmethod
    def validate_device_ids(cls, value: List[int]) -> List[int]:
        if any(device_id < 0 or device_id > 63 for device_id in value):
            raise ValueError("device_ids must be between 0 and 63")
        if len(set(value)) != len(value):
            raise ValueError("device_ids must be unique")
        return value


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    external_task_id: str = Field(min_length=1, max_length=256)
    task_type: str = Field(default="model_deployment", min_length=1)
    scenario: str = Field(default="llm", min_length=1)
    chip: str = Field(default="Ascend_910B", min_length=1)
    chip_count: int = Field(default=1, ge=1, le=64)
    image: ImageSpec
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    benchmark: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("chip")
    @classmethod
    def validate_chip(cls, value: str) -> str:
        normalized = value.lower().replace("-", "_")
        if normalized not in {"ascend_910b", "huawei_910b"}:
            raise ValueError("the execution service currently supports Ascend_910B only")
        return value

    @model_validator(mode="after")
    def validate_device_count(self):
        if self.resources.device_ids and len(self.resources.device_ids) != self.chip_count:
            raise ValueError("chip_count must match the number of device_ids")
        return self


JobStatus = Literal[
    "queued",
    "preparing",
    "starting_container",
    "running",
    "collecting",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
]


class JobView(BaseModel):
    job_id: str
    external_task_id: str
    status: JobStatus
    progress: int
    created_at: str
    updated_at: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    container: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)


class LogsView(BaseModel):
    job_id: str
    logs: str
