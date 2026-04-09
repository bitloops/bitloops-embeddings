from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    dimensions: int
    embeddings: list[list[float]]
    runtime: RuntimeInfo


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    model_id: str
    dimensions: int
    runtime_version: str


class DescribeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    dimensions: int
    backend_name: str
    cache_dir: str
    runtime_version: str


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str]

