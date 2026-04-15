from __future__ import annotations

import bitloops_local_embeddings.__main__ as main_module
import bitloops_local_embeddings.cli as cli_module


def test_run_calls_freeze_support_before_cli(monkeypatch) -> None:
    calls: list[str] = []

    def fake_freeze_support() -> None:
        calls.append("freeze_support")

    def fake_main() -> None:
        calls.append("main")

    monkeypatch.setattr(main_module.multiprocessing, "freeze_support", fake_freeze_support)
    monkeypatch.setattr(cli_module, "main", fake_main)

    main_module.run()

    assert calls == ["freeze_support", "main"]
