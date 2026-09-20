"""Audio conditioning before transcription.

Deliberately conservative. Whisper was trained on large amounts of noisy,
band-limited, real-world audio, and aggressive denoising routinely makes its
output *worse* -- spectral subtraction leaves musical artefacts that the model
has never seen, where it handles ordinary hiss and telephone bandwidth well.

So this does the things that reliably help and stops:

- **DC offset removal.** A non-zero mean wastes headroom and skews the mel
  spectrogram.
- **High-pass at 80 Hz.** Below the telephone passband there is nothing but
  rumble, handling noise and mains hum.
- **Peak-safe loudness normalisation.** 911 recordings vary hugely in level;
  Whisper's mel front end is not level-invariant in practice.
- **Optional gentle spectral gate**, off by default, for recordings that are
  genuinely noise-dominated.

Everything returns measurements alongside the audio, so the effect can be
checked rather than assumed.
"""
import numpy as np
from scipy import signal


def _db(x):
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-12))


def estimate_snr_db(y, sr, frame_ms=25):
    """Crude SNR: loud frames over quiet frames.

    Not a calibrated measurement -- it is a comparable number for deciding
    whether a recording is noise-dominated, and for reporting whether cleaning
    changed anything.
    """
    n = max(1, int(sr * frame_ms / 1000))
    frames = y[: len(y) // n * n].reshape(-1, n)
    if frames.size == 0:
        return 0.0
    energy = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    quiet = np.percentile(energy, 10)
    loud = np.percentile(energy, 90)
    return float(20.0 * np.log10(loud / max(quiet, 1e-12)))


def highpass(y, sr, cutoff_hz=80.0, order=4):
    if sr <= 2 * cutoff_hz:
        return y
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def normalise(y, target_dbfs=-20.0, peak_ceiling=0.97):
    """Bring RMS to a target level without clipping."""
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    if rms < 1e-9:
        return y
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    peak = float(np.max(np.abs(y))) or 1.0
    gain = min(gain, peak_ceiling / peak)
    return (y * gain).astype(np.float32)


def spectral_gate(y, sr, noise_seconds=0.5, reduction_db=9.0):
    """Gentle spectral gate, estimating the noise floor from the quietest part.

    Capped at a modest reduction on purpose: deep subtraction produces the
    warbling artefacts that hurt ASR more than the noise did.
    """
    n_fft, hop = 1024, 256
    f, t, Z = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag, phase = np.abs(Z), np.angle(Z)

    frames = max(1, int(noise_seconds * sr / hop))
    energy = mag.sum(axis=0)
    quietest = np.argsort(energy)[:frames]
    noise = np.median(mag[:, quietest], axis=1, keepdims=True)

    floor = 10.0 ** (-reduction_db / 20.0)
    cleaned = np.maximum(mag - noise, mag * floor)
    _, out = signal.istft(cleaned * np.exp(1j * phase), fs=sr,
                          nperseg=n_fft, noverlap=n_fft - hop)
    out = out[: len(y)]
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out.astype(np.float32)


def clean_audio(y, sr, denoise=False, target_dbfs=-20.0, highpass_hz=80.0):
    """Condition audio for ASR. Returns (audio, report)."""
    y = np.asarray(y, dtype=np.float32)
    report = {
        "input_rms_dbfs": round(_db(y), 1),
        "input_peak": round(float(np.max(np.abs(y))) if y.size else 0.0, 3),
        "input_snr_db": round(estimate_snr_db(y, sr), 1),
        "denoised": False,
    }

    y = y - float(np.mean(y))            # DC offset
    y = highpass(y, sr, highpass_hz)

    if denoise:
        y = spectral_gate(y, sr)
        report["denoised"] = True

    y = normalise(y, target_dbfs)

    report["output_rms_dbfs"] = round(_db(y), 1)
    report["output_peak"] = round(float(np.max(np.abs(y))) if y.size else 0.0, 3)
    report["output_snr_db"] = round(estimate_snr_db(y, sr), 1)
    return y, report
