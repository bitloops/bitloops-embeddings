from __future__ import annotations

import multiprocessing


def run() -> None:
    # PyInstaller requires freeze_support() in the frozen entrypoint so
    # multiprocessing helper processes do not recurse into the CLI parser.
    multiprocessing.freeze_support()
    from bitloops_embeddings.cli import main

    main()

if __name__ == "__main__":
    run()
