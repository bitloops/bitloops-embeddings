from __future__ import annotations

import sys

from tests.support import register_fake_model


def main() -> None:
    register_fake_model()
    forwarded_arguments = sys.argv[1:]

    from bitloops_local_embeddings.cli import main as cli_main

    sys.argv = [
        "bitloops-local-embeddings",
        "daemon",
        "--model",
        "bge-m3",
        *forwarded_arguments,
    ]
    cli_main()


if __name__ == "__main__":
    main()
