"""Make pip-installed CUDA libraries visible to CTranslate2.

torch ships CUDA as `nvidia-*` wheels under `site-packages/nvidia/*/lib`, and
loads them itself. CTranslate2 -- the backend behind faster-whisper -- links
against `libcublas.so.12` and friends through the ordinary dynamic loader,
which never looks there. The result is

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

on a machine where CUDA demonstrably works, because torch is using the very
libraries CTranslate2 cannot find.

Exporting `LD_LIBRARY_PATH` fixes it, but only if it is set before the process
starts, which makes `python app.py` fail and `LD_LIBRARY_PATH=... python app.py`
succeed -- exactly the kind of undocumented setup step this project has been
burned by. Preloading with `RTLD_GLOBAL` instead puts the symbols in the global
namespace at import time, so a later `dlopen` resolves against them.

Import this before `faster_whisper`. Failure is not fatal: the caller falls
back to CPU.
"""
import ctypes
import glob
import os
import sys

# Order matters: cublas depends on cublasLt, cudnn on the cuda runtime.
_PREFERRED = [
    "cuda_runtime", "cublas", "cudnn", "cufft", "curand",
    "cusolver", "cusparse", "nvrtc", "cuda_nvrtc",
]


def _candidate_dirs():
    seen, dirs = set(), []
    for base in {os.path.dirname(os.path.dirname(m.__file__))
                 for name, m in sys.modules.items()
                 if name == "torch" and getattr(m, "__file__", None)} or \
                {p for p in sys.path if p.endswith("site-packages")}:
        root = os.path.join(base, "nvidia")
        if not os.path.isdir(root):
            continue
        for name in _PREFERRED + sorted(os.listdir(root)):
            lib = os.path.join(root, name, "lib")
            if os.path.isdir(lib) and lib not in seen:
                seen.add(lib)
                dirs.append(lib)
    return dirs


def preload(verbose=False):
    """Load every CUDA .so we can find into the global symbol namespace.

    Returns the list of libraries loaded. Errors are swallowed on purpose --
    a missing optional library is not a reason to fail, and the caller decides
    what to do when CUDA turns out to be unusable.
    """
    loaded = []
    for directory in _candidate_dirs():
        for path in sorted(glob.glob(os.path.join(directory, "lib*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(os.path.basename(path))
            except OSError as exc:
                if verbose:
                    print(f"  skip {os.path.basename(path)}: {exc}")
    return loaded


def cuda_usable():
    """Whether CTranslate2 can actually run on the GPU, after preloading."""
    try:
        preload()
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
