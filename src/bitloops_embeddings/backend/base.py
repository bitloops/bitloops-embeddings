from __future__ import annotations

from typing import Protocol


class EmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def load(self) -> None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

