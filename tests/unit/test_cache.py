from __future__ import annotations

from pathlib import Path

from bitloops_embeddings.cache import CACHE_ENV_VAR, resolve_cache_dir


def test_explicit_cache_dir_has_highest_priority(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-cache"
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env-cache"))

    assert resolve_cache_dir(explicit) == explicit


def test_env_cache_dir_is_used_when_present(monkeypatch, tmp_path: Path) -> None:
    env_cache_dir = tmp_path / "env-cache"
    monkeypatch.setenv(CACHE_ENV_VAR, str(env_cache_dir))

    assert resolve_cache_dir() == env_cache_dir


def test_platform_default_cache_dir_is_used_as_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "bitloops_embeddings.cache.user_cache_dir",
        lambda app_name: str(tmp_path / app_name),
    )

    assert resolve_cache_dir() == tmp_path / "bitloops-embeddings"
