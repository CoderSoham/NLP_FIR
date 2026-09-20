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
