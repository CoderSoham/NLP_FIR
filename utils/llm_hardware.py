"""Recommend a local LLM that actually fits the machine it will run on.

Picking a default model for someone else's hardware is guesswork, so this
inspects the machine and offers options instead. Sizes are for the weights at
the stated quantisation, with headroom left for the ASR model, which shares the
GPU.
"""
import os

# (id, params, VRAM GB at float16, RAM GB on CPU, note)
#
# These are float16 weights, which is what the loader actually uses. An earlier
# version of this table quoted 4-bit figures while the loader ran float16, so it
# recommended a 3.8B model for a 6 GB card and the load failed. Advertise what
# the code does, not what it could do with a quantiser it does not use.
CATALOGUE = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B", 1.1, 3,
     "Fits anywhere. Use only when nothing larger will load."),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B", 3.2, 7,
     "The practical floor for reliable JSON extraction. Default on a 6 GB card."),
    ("Qwen/Qwen2.5-3B-Instruct", "3B", 6.2, 13,
     "Clearly better at ambiguous calls. Needs a 8 GB card."),
    ("microsoft/Phi-3-mini-4k-instruct", "3.8B", 7.6, 16,
     "Strong at structured extraction; shorter context. 10 GB card."),
    ("Qwen/Qwen2.5-7B-Instruct", "7B", 15.2, 30,
     "Noticeably better reasoning. 16 GB card, or run it on CPU."),
    ("Qwen/Qwen2.5-14B-Instruct", "14B", 29.0, 60,
     "Approaching useful judgement on ambiguous calls. 32 GB card."),
]


def probe():
    """What hardware is available. Never raises."""
    info = {"cuda": False, "vram_gb": 0.0, "gpu": None, "ram_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(cuda=True, gpu=props.name,
                        vram_gb=round(props.total_memory / 1e9, 1))
    except Exception:
        pass
    try:
        info["ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        pass
    return info


def recommend(info=None, asr_reserve_gb=0.0):
    """Return (recommended_id, options) for this machine.

    `asr_reserve_gb` defaults to 0 because `utils.gpu` evicts the ASR model
    before the LLM loads -- only one large model is GPU-resident at a time. Pass
    a reserve if you set `GPU_EXCLUSIVE=0` and want them co-resident.

    A further ~0.8 GB is held back for the display and CUDA context, which are
    real and were the difference between "fits on paper" and an OOM.
    """
    info = info or probe()
    budget = max(0.0, info["vram_gb"] - asr_reserve_gb - 0.8) if info["cuda"] else 0.0

    options = []
    for model_id, params, vram, ram, note in CATALOGUE:
        if info["cuda"]:
            fits = vram <= budget
            need = f"{vram} GB VRAM"
        else:
            fits = ram <= info["ram_gb"]
            need = f"{ram} GB RAM"
        options.append({"model": model_id, "params": params, "needs": need,
                        "fits": fits, "note": note})

    fitting = [o for o in options if o["fits"]]
    recommended = fitting[-1]["model"] if fitting else CATALOGUE[0][0]
    return recommended, options


def describe():
    """Human-readable summary, for the CLI helper and the logs."""
    info = probe()
    recommended, options = recommend(info)
    lines = []
    if info["cuda"]:
        lines.append(f"GPU: {info['gpu']} ({info['vram_gb']} GB VRAM), "
                     f"{info['ram_gb']} GB system RAM")
    else:
        lines.append(f"No CUDA GPU detected. {info['ram_gb']} GB system RAM")
    lines.append("")
    for o in options:
        mark = "recommended" if o["model"] == recommended else ("ok" if o["fits"] else "too large")
        lines.append(f"  [{mark:11}] {o['model']:38} {o['params']:>5}  needs {o['needs']:<12} {o['note']}")
    lines.append("")
    lines.append(f"Set LOCAL_LLM_MODEL to choose. Default: {recommended}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
