"""The two requirements files have to agree with each other.

CI installs `requirements-dev.txt` for the fast suite and dry-runs
`requirements.txt` separately. That split is what keeps the suite fast, and it
is also a place for the two to drift apart silently: adding fpdf2 to the real
requirements and not the dev ones turned every CI job red on a change that was
green locally, because the local venv had both.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)\s*$", re.M)


def pins(filename):
    with open(os.path.join(REPO_ROOT, filename), encoding="utf-8") as fh:
        return {name.lower(): version for name, version in PIN.findall(fh.read())}


def test_shared_dependencies_are_pinned_to_the_same_version():
    runtime, dev = pins("requirements.txt"), pins("requirements-dev.txt")
    drifted = {name: (dev[name], runtime[name]) for name in dev.keys() & runtime.keys()
               if dev[name] != runtime[name]}
    assert not drifted, f"dev vs runtime pin drift: {drifted}"


def test_everything_the_fast_suite_imports_is_a_dev_dependency():
    """A module importable without torch has its dependencies tested, so they
    are test dependencies whether or not anyone remembered to say so."""
    dev = pins("requirements-dev.txt")
    for package in ("fpdf2", "matplotlib", "flask", "pytest"):
        assert package in dev, f"{package} is imported by the fast suite"


# Three requirements carry a range rather than a pin, each for a reason:
# numba is a ceiling for librosa compatibility, and the other two are floors on
# packages whose newer releases are wanted. Listing them here means a fourth
# range has to be argued for rather than typed.
DELIBERATE_RANGES = {"numba", "accelerate", "anthropic"}


def test_every_runtime_requirement_carries_a_version_constraint():
    """An unconstrained dependency makes the build unreproducible, and this
    project has already lost a day to pip resolving an upgrade on its own --
    `pip install bitsandbytes` took torch from 2.2.2 to 2.14.0."""
    with open(os.path.join(REPO_ROOT, "requirements.txt"), encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh
                 if ln.strip() and not ln.strip().startswith("#")]

    loose = []
    for line in lines:
        if " @ " in line:                       # a direct URL is exact already
            continue
        name = re.split(r"[=<>!~ ]", line, 1)[0].lower()
        if "==" in line:
            continue
        if any(op in line for op in ("<", ">", "~=")):
            if name not in DELIBERATE_RANGES:
                loose.append(line)
            continue
        loose.append(line)                      # no constraint at all

    assert not loose, f"unconstrained or newly-ranged requirements: {loose}"


def test_the_whole_suite_collects_without_torch():
    """CI installs requirements-dev.txt, which has no torch. A test that
    reaches into utils.audio_utils passes on a development machine and fails
    on every CI job -- which is exactly what happened when three comparison
    tests imported it, and is why utils/compare.py exists.

    Collection is enough: an import-time dependency shows up there.
    """
    import subprocess
    import sys
    import textwrap

    # A meta-path hook is used rather than uninstalling anything, so the
    # check costs a second and needs no separate environment.
    program = textwrap.dedent("""
        import sys

        BLOCKED = ("torch", "transformers", "librosa", "spacy",
                   "sentence_transformers", "faster_whisper", "sklearn")

        class Blocker:
            # find_spec, not find_module: the latter was removed in 3.12, so
            # a blocker written against it silently does nothing and the
            # guard passes without guarding anything.
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{name} is not installed in CI")
                return None

        sys.meta_path.insert(0, Blocker())
        import pytest
        sys.exit(pytest.main(["--collect-only", "-q", "tests/"]))
    """)
    result = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        "the suite does not collect without the ML stack:\n"
        + result.stdout[-3000:] + result.stderr[-2000:])
