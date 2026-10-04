"""Release gates must reject mixed, incomplete or modified distribution sets."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_artifacts", ROOT / ".github/scripts/release_artifacts.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
VERSION = "2.1.7"
SHA = "a" * 40


def metadata(version=VERSION):
    return f"Metadata-Version: 2.1\nName: UQPyL\nVersion: {version}\n".encode()


def makeWheel(path, version=VERSION):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"uqpyl-{version}.dist-info/METADATA", metadata(version))


@pytest.fixture
def candidate(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    for python in release.PYTHONS:
        for platform in ("manylinux_2_28_x86_64", "win_amd64", "macosx_10_13_x86_64", "macosx_11_0_arm64"):
            makeWheel(dist / f"uqpyl-{VERSION}-{python}-{python}-{platform}.whl")
    with tarfile.open(dist / f"uqpyl-{VERSION}.tar.gz", "w:gz") as archive:
        data = metadata()
        info = tarfile.TarInfo(f"uqpyl-{VERSION}/PKG-INFO")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    release.createManifest(tmp_path, VERSION, SHA, "42", "owner/repo")
    return tmp_path


def verify(candidate, **overrides):
    args = dict(version=VERSION, sha=SHA, runId="42", repository="owner/repo")
    args.update(overrides)
    return release.verifyManifest(candidate, **args)


def testCompleteRelease(candidate):
    assert len(verify(candidate)["files"]) == 21


@pytest.mark.parametrize(
    "field,value", [("version", "2.1.6"), ("sha", "b" * 40), ("runId", "43"), ("repository", "other/repo")]
)
def testRejectWrongIdentity(candidate, field, value):
    with pytest.raises(ValueError, match="selected source"):
        verify(candidate, **{field: value})


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def testMissingDistribution(candidate, kind):
    next((candidate / "dist").glob("*.whl" if kind == "wheel" else "*.tar.gz")).unlink()
    with pytest.raises(ValueError, match="Expected 20"):
        verify(candidate)


def testChangedWheelBytes(candidate):
    wheel = next((candidate / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("changed.txt", "unexpected change")
    with pytest.raises(ValueError, match="hashes changed"):
        verify(candidate)


def testChangedEmbeddedVersion(candidate):
    makeWheel(next((candidate / "dist").glob("*.whl")), "2.1.6")
    with pytest.raises(ValueError, match="metadata"):
        verify(candidate)


def testUnknownFile(candidate):
    (candidate / "dist/token.txt").write_text("not a distribution")
    with pytest.raises(ValueError, match="Unexpected file"):
        verify(candidate)


def testDuplicateTarget(candidate):
    makeWheel(candidate / "dist" / f"uqpyl-{VERSION}-cp312-cp312-manylinux_2_17_x86_64.whl")
    with pytest.raises(ValueError, match="Duplicate wheel"):
        verify(candidate)


@pytest.mark.parametrize("tag", ["linux_x86_64", "any", "macosx_11_0_universal2", "win32"])
def testUnpublishableOrUnplannedPlatform(candidate, tag):
    makeWheel(candidate / "dist" / f"uqpyl-{VERSION}-cp312-cp312-{tag}.whl")
    with pytest.raises(ValueError, match="Unexpected wheel platform"):
        verify(candidate)


def testDualManylinuxTagsCountOnce(candidate):
    old = next((candidate / "dist").glob("*-cp312-cp312-manylinux*.whl"))
    old.rename(old.with_name(old.name.replace("manylinux_2_28_x86_64", "manylinux_2_17_x86_64.manylinux2014_x86_64")))
    assert len(release.inventory(candidate, VERSION)) == 21


@pytest.mark.parametrize("tag", ["2.1.7", "v2.1.6", "refs/tags/v2.1.7", "v2.1.7/extra"])
def testWrongReleaseTag(tag):
    with pytest.raises(ValueError, match="Release tag"):
        release.checkTag(tag, VERSION)


def testSourceVersionConsistency():
    version = release.projectVersion(ROOT)
    release.checkTag(f"v{version}", version)


def testWorkflowsHaveNoRebuildInPublisher():
    # BaseLoader keeps YAML's `on` key literal instead of treating it as boolean.
    ci = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    publish = yaml.load((ROOT / ".github/workflows/build.yml").read_text(), Loader=yaml.BaseLoader)
    matrix = ci["jobs"]["tests"]["strategy"]["matrix"]
    assert set(matrix["python"]) == release.PYTHONS
    assert {target["id"] for target in matrix["target"]} == release.PLATFORMS
    assert len(matrix["python"]) * len(matrix["target"]) == len(release.EXPECTED)
    assert ci["jobs"]["tests"]["needs"] == "style"
    assert ci["jobs"]["sdist"]["needs"] == "style"
    assert set(ci["jobs"]["candidate"]["needs"]) == {"style", "tests", "sdist"}
    assert publish["on"]["workflow_dispatch"]["inputs"]["publish"]["default"] == "false"
    assert publish["jobs"]["publish"]["needs"] == "prepare"
    commands = [step.get("run", "") for job in publish["jobs"].values() for step in job["steps"]]
    assert not any("cibuildwheel" in command or "-m build" in command for command in commands)
    for job in publish["jobs"].values():
        downloads = [step for step in job["steps"] if step.get("uses", "").startswith("actions/download-artifact@")]
        assert downloads and all("run-id" in step["with"] and "github-token" in step["with"] for step in downloads)
