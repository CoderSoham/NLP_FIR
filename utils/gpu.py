"""One large model on the GPU at a time.

A 6 GB card cannot hold `large-v3-turbo` and a multi-billion-parameter instruct
model at once. Loading both is not slow, it fails:

    RuntimeError: CUDA failed with error out of memory

and it fails on the *second* request, after the first has already succeeded --
the worst possible shape for a bug, because a smoke test passes.

The stages run sequentially within a request, so residency can be sequential
too. Each GPU-resident component registers a releaser; acquiring a slot evicts
everyone else. Reloading costs a few seconds from the page cache, which is a
better trade than an unpredictable OOM.

Set `GPU_EXCLUSIVE=0` on a card with room to spare to keep everything resident.
"""
import os
import threading

_LOCK = threading.Lock()
_RELEASERS = {}
_CURRENT = None


def exclusive_mode():
    return os.environ.get("GPU_EXCLUSIVE", "1") == "1"


def register(name, releaser):
    """Register a component's unload callback under a stable name."""
    with _LOCK:
        _RELEASERS[name] = releaser


def free_vram_gb():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return round(free / 1e9, 2)
    except Exception:
        return 0.0


def acquire(name):
    """Claim the GPU for `name`, evicting other registered components.

    Returns the names evicted, so a caller can log why a reload happened.
    """
    global _CURRENT
    if not exclusive_mode():
        return []
    with _LOCK:
        if _CURRENT == name:
            return []
        evicted = []
        for other, releaser in _RELEASERS.items():
            if other == name:
                continue
            try:
                if releaser():
                    evicted.append(other)
            except Exception:
                pass
        _CURRENT = name
    if evicted:
        empty_cache()
    return evicted


def empty_cache():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def reset():
    """Forget the current holder. For tests."""
    global _CURRENT
    with _LOCK:
        _CURRENT = None
