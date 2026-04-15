from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "bitloops-local-embeddings"
VERSION_FILE = ROOT / "src" / "bitloops_local_embeddings" / "version.py"
ENTRYPOINT = ROOT / "src" / "bitloops_local_embeddings" / "__main__.py"
PYINSTALLER_COLLECT_SUBMODULES = [
    "sentence_transformers",
    "transformers.models",
]
PYINSTALLER_COLLECT_DATA = [
    "sentence_transformers",
    "transformers",
]
PYINSTALLER_COPY_METADATA = [
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "torch",
]
PYINSTALLER_EXCLUDED_MODULES = [
    "pytest",
    "IPython",
    "jedi",
    "matplotlib",
    "notebook",
    "pandas",
    "tensorboard",
    "tensorflow",
    "torchvision",
    "torchaudio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and package a native release archive.")
    parser.add_argument("--target", default=None, help="Release target triple.")
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
    archive_dir = Path(args.archive_dir).resolve()

    archive_path, bundle_dir, bundle_executable = build_release(
        version=version,
        target=target,
        archive_dir=archive_dir,
    )

    print(f"Created {archive_path}")
    if args.github_output:
        write_github_outputs(
            Path(args.github_output),
            {
                "archive_path": archive_path.as_posix(),
                "bundle_dir": bundle_dir.as_posix(),
                "bundle_executable": bundle_executable.as_posix(),
                "version": version,
                "target": target,
            },
        )


def build_release(*, version: str, target: str, archive_dir: Path) -> tuple[Path, Path, Path]:
    build_root = ROOT / "build"
    pyinstaller_dist = build_root / "pyinstaller-dist" / target
    pyinstaller_work = build_root / "pyinstaller-work" / target
    package_root = build_root / "package"
    archive_root_name = f"{PACKAGE_NAME}-v{version}-{target}"
    staging_dir = package_root / archive_root_name
    bundle_dir = staging_dir / PACKAGE_NAME
    bundle_executable = bundle_dir / executable_name_for_target(target)

    shutil.rmtree(pyinstaller_dist, ignore_errors=True)
    shutil.rmtree(pyinstaller_work, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    pyinstaller_dist.mkdir(parents=True, exist_ok=True)
    pyinstaller_work.mkdir(parents=True, exist_ok=True)

    run_pyinstaller(
        pyinstaller_dist=pyinstaller_dist,
        pyinstaller_work=pyinstaller_work,
        codesign_identity=os.environ.get("APPLE_SIGNING_IDENTITY"),
    )

    shutil.copytree(pyinstaller_dist / PACKAGE_NAME, bundle_dir)
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
    elif archive_extension == ".tar.xz":
        with tarfile.open(archive_path, "w:xz") as archive:
            archive.add(staging_dir, arcname=archive_root_name)
    else:
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_dir, arcname=archive_root_name)

    return archive_path, bundle_dir, bundle_executable


def run_pyinstaller(
    *,
    pyinstaller_dist: Path,
    pyinstaller_work: Path,
    codesign_identity: str | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        PACKAGE_NAME,
        "--distpath",
        str(pyinstaller_dist),
        "--workpath",
        str(pyinstaller_work),
        "--paths",
        str(ROOT / "src"),
    ]

    if sys.platform == "darwin" and codesign_identity:
        command.extend(["--codesign-identity", codesign_identity])

    for module_name in PYINSTALLER_COLLECT_SUBMODULES:
        command.extend(["--collect-submodules", module_name])

    for module_name in PYINSTALLER_COLLECT_DATA:
        command.extend(["--collect-data", module_name])

    for package_name in PYINSTALLER_COPY_METADATA:
        command.extend(["--recursive-copy-metadata", package_name])

    for module_name in PYINSTALLER_EXCLUDED_MODULES:
        command.extend(["--exclude-module", module_name])

    command.append(str(ENTRYPOINT))

    subprocess.run(command, check=True, cwd=ROOT)


def read_version() -> str:
    for line in VERSION_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"')
    raise RuntimeError("Unable to determine the project version.")


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
