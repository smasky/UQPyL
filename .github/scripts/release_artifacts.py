"""Validate release identity and the complete tested binary distribution set.

This script never builds or uploads distributions. Use Python 3.12 or newer.
"""

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename

PYTHONS = {f"cp3{minor}" for minor in range(10, 15)}
PLATFORMS = {"linux-x86_64", "windows-amd64", "macos-x86_64", "macos-arm64"}
EXPECTED = {(python, platform) for python in PYTHONS for platform in PLATFORMS}


def projectVersion(root):
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    tree = ast.parse((root / "UQPyL/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if versions != [version]:
        raise ValueError("Package and project versions disagree.")
    return version


def checkTag(tag, version):
    if tag != f"v{version}":
        raise ValueError(f"Release tag must equal v{version}.")


def checkMetadata(data, version):
    metadata = BytesParser().parsebytes(data)
    if canonicalize_name(metadata.get("Name", "")) != "uqpyl" or metadata.get("Version") != version:
        raise ValueError("Distribution metadata name/version does not match the project.")


def wheelTarget(tags):
    targets = set()
    for tag in tags:
        if tag.interpreter not in PYTHONS or tag.abi != tag.interpreter:
            raise ValueError("Unexpected Python/ABI in wheel.")
        platform = tag.platform
        if platform.startswith("manylinux") and platform.endswith("_x86_64"):
            target = "linux-x86_64"
        elif platform == "win_amd64":
            target = "windows-amd64"
        elif platform.startswith("macosx_") and platform.endswith("_x86_64"):
            target = "macos-x86_64"
        elif platform.startswith("macosx_") and platform.endswith("_arm64"):
            target = "macos-arm64"
        else:
            raise ValueError(f"Unexpected wheel platform: {platform}")
        targets.add((tag.interpreter, target))
    if len(targets) != 1:
        raise ValueError("Wheel must cover exactly one matrix target.")
    return targets.pop()


def inventory(directory, version):
    dist = directory / "dist"
    files = sorted(dist.iterdir())
    targets = set()
    sdistCount = 0
    entries = []
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Distribution directory must contain regular files only.")
        if path.suffix == ".whl":
            name, fileVersion, _, tags = parse_wheel_filename(path.name)
            target = wheelTarget(tags)
            if target in targets:
                raise ValueError(f"Duplicate wheel target: {target}")
            targets.add(target)
            with zipfile.ZipFile(path) as archive:
                metadataPaths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadataPaths) != 1:
                    raise ValueError("Wheel must contain exactly one METADATA file.")
                checkMetadata(archive.read(metadataPaths[0]), version)
        elif path.name.endswith(".tar.gz"):
            name, fileVersion = parse_sdist_filename(path.name)
            sdistCount += 1
            with tarfile.open(path) as archive:
                metadataPaths = [
                    entry
                    for entry in archive.getmembers()
                    if entry.name.endswith("/PKG-INFO") and entry.name.count("/") == 1 and entry.isfile()
                ]
                if len(metadataPaths) != 1:
                    raise ValueError("Source archive must contain one root PKG-INFO.")
                checkMetadata(archive.extractfile(metadataPaths[0]).read(), version)
        else:
            raise ValueError(f"Unexpected file in distribution directory: {path.name}")
        if name != "uqpyl" or str(fileVersion) != version:
            raise ValueError("Distribution filename name/version does not match the project.")
        entries.append(
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
        )
    if targets != EXPECTED or sdistCount != 1:
        raise ValueError(f"Expected 20 wheel targets and one sdist; missing={sorted(EXPECTED - targets)}.")
    return entries


def createManifest(directory, version, sha, runId, repository):
    result = {
        "schema": 1,
        "version": version,
        "commit_sha": sha,
        "run_id": str(runId),
        "repository": repository,
        "files": inventory(directory, version),
    }
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verifyManifest(directory, version, sha, runId, repository):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = {"schema": 1, "version": version, "commit_sha": sha, "run_id": str(runId), "repository": repository}
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"Candidate {name} does not match the selected source.")
    if manifest.get("files") != inventory(directory, version):
        raise ValueError("Candidate file list, sizes or hashes changed.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["version", "manifest", "verify"])
    parser.add_argument("--directory", type=Path, default=Path("candidate"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    version = projectVersion(root)
    if args.command in {"version", "verify"}:
        checkTag(os.environ["RELEASE_TAG"], version)
    if args.command != "version":
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        repository = os.environ["GITHUB_REPOSITORY"]
        if args.command == "manifest":
            if sha != os.environ["GITHUB_SHA"]:
                raise ValueError("Checkout does not match the CI source SHA.")
            createManifest(args.directory, version, sha, os.environ["GITHUB_RUN_ID"], repository)
        else:
            verifyManifest(args.directory, version, sha, os.environ["SOURCE_RUN_ID"], repository)
    print(f"Release {version}: {args.command} validation passed.")


if __name__ == "__main__":
    main()
