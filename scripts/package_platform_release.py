from __future__ import annotations

import argparse
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "bitloops-platform-embeddings"
CRATE_MANIFEST = ROOT / "crates" / PACKAGE_NAME / "Cargo.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Rust platform embeddings binary.")
    parser.add_argument("--target", default=None, help="Release target triple.")
    parser.add_argument(
        "--binary-path",
        default=None,
        help="Path to the compiled binary. Defaults to Cargo's native release path.",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(ROOT / "build" / "artifacts"),
        help="Directory where the final archive should be written.",
    )
    parser.add_argument(
        "--github-output",
        default=None,
        help="Optional path to the GitHub Actions output file.",
    )
    args = parser.parse_args()

    version = read_version()
    target = args.target or detect_target()
    binary_path = Path(args.binary_path).resolve() if args.binary_path else default_binary_path(target)
    archive_dir = Path(args.archive_dir).resolve()

    archive_path, bundle_executable = build_release(
        version=version,
        target=target,
        archive_dir=archive_dir,
        binary_path=binary_path,
    )

    print(f"Created {archive_path}")
    if args.github_output:
        write_github_outputs(
            Path(args.github_output),
            {
                "archive_path": archive_path.as_posix(),
                "bundle_executable": bundle_executable.as_posix(),
                "version": version,
                "target": target,
            },
        )


def build_release(*, version: str, target: str, archive_dir: Path, binary_path: Path) -> tuple[Path, Path]:
    if not binary_path.is_file():
        raise RuntimeError(f"Binary path does not exist: {binary_path}")

    build_root = ROOT / "build"
    package_root = build_root / "platform-package"
    archive_root_name = f"{PACKAGE_NAME}-v{version}-{target}"
    staging_dir = package_root / archive_root_name
    bundle_executable = staging_dir / executable_name_for_target(target)

    shutil.rmtree(staging_dir, ignore_errors=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(binary_path, bundle_executable)
    shutil.copy2(ROOT / "README.md", staging_dir / "README.md")
    shutil.copy2(ROOT / "LICENSE", staging_dir / "LICENSE")

    archive_extension = archive_extension_for_target(target)
    archive_path = archive_dir / f"{archive_root_name}{archive_extension}"
    if archive_path.exists():
        archive_path.unlink()

    if archive_extension == ".zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staging_dir.rglob("*"):
                archive.write(path, arcname=path.relative_to(package_root))
    else:
        with tarfile.open(archive_path, "w:xz") as archive:
            archive.add(staging_dir, arcname=archive_root_name)

    return archive_path, bundle_executable


def read_version() -> str:
    for line in CRATE_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("Unable to determine the platform runtime version.")


def detect_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    mapping = {
        ("darwin", "x86_64"): "x86_64-apple-darwin",
        ("darwin", "arm64"): "aarch64-apple-darwin",
        ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("windows", "amd64"): "x86_64-pc-windows-msvc",
    }
    try:
        return mapping[(system, machine)]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported local packaging target for system={system} machine={machine}."
        ) from exc


def default_binary_path(target: str) -> Path:
    return ROOT / "target" / target / "release" / executable_name_for_target(target)


def executable_name_for_target(target: str) -> str:
    if target.endswith("windows-msvc"):
        return f"{PACKAGE_NAME}.exe"
    return PACKAGE_NAME


def archive_extension_for_target(target: str) -> str:
    if target.endswith("windows-msvc") or target.endswith("apple-darwin"):
        return ".zip"
    return ".tar.xz"


def write_github_outputs(output_path: Path, outputs: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8") as file_handle:
        for key, value in outputs.items():
            file_handle.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
