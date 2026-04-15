from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir


CACHE_ENV_VAR = "BITLOOPS_LOCAL_EMBEDDINGS_CACHE_DIR"


def resolve_cache_dir(explicit_cache_dir: Optional[Path] = None) -> Path:
    if explicit_cache_dir is not None:
        return explicit_cache_dir.expanduser()

    env_cache_dir = os.getenv(CACHE_ENV_VAR)
    if env_cache_dir:
        return Path(env_cache_dir).expanduser()

    return Path(user_cache_dir("bitloops-local-embeddings")).expanduser()


def ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
