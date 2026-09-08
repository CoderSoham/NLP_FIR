"""Make a bundled ffmpeg discoverable without a system package.

Whisper shells out to `ffmpeg` by name, so it has to be on PATH. Requiring it
from apt or brew means the quick start cannot be completed without root, which
is a real barrier on managed machines and in slim containers.

`imageio-ffmpeg` ships a static build as an ordinary wheel. This puts its
directory on PATH, and does nothing if a system ffmpeg is already there --
a distro build is preferable when it exists.
"""
import os
import shutil


def ensure_ffmpeg_on_path():
    """Return the ffmpeg that will be used, or None if there is none."""
    existing = shutil.which("ffmpeg")
    if existing:
        return existing
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    if not exe or not os.path.isfile(exe):
        return None
    os.environ["PATH"] = os.path.dirname(exe) + os.pathsep + os.environ.get("PATH", "")
    # Whisper invokes the bare name, and the bundled binary is not called
    # "ffmpeg", so expose it under the expected name next to itself.
    alias = os.path.join(os.path.dirname(exe), "ffmpeg")
    if not os.path.exists(alias):
        try:
            os.symlink(exe, alias)
        except OSError:
            return exe
    return shutil.which("ffmpeg") or exe
