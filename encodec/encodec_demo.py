"""
EnCodec — RVQ Perceptual Codec Demo
====================================
Uses Meta's pretrained EnCodec model to encode and decode audio at multiple
bitrates using Residual Vector Quantization (RVQ).

Each bitrate corresponds to a different number of RVQ codebooks.
Lower codebook counts → more compression → more artefacts.

Install
-------
    pip install torch torchaudio encodec torchcodec gradio matplotlib

Run
---
    python encodec_demo.py
"""

import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import gradio as gr
from encodec import EncodecModel

matplotlib.use("Agg")   # non-interactive backend for server use
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# ─── EnCodec constants ────────────────────────────────────────────────────────
# EnCodec architecture:
#   - Encoder: strided conv (total stride 320) → continuous embeddings
#   - RVQ    : N codebooks of 1024 entries (10 bits each)
#   - Decoder: mirrored transposed convs
#
# Bitrate formula:
#   bw_kbps = N_codebooks × frame_rate × bits_per_codebook / 1000
#   frame_rate = sample_rate / encoder_stride
#
ENCODER_STRIDE = 320
BITS_PER_CODEBOOK = 10          # log2(1024)
ALL_BANDWIDTHS = [3.0, 6.0, 12.0, 24.0]   # kbps (1.5 listed by model but unsupported)

def frame_rate(sample_rate: int) -> float:
    return sample_rate / ENCODER_STRIDE

def n_codebooks(bw_kbps: float, sample_rate: int) -> int:
    return int(bw_kbps * 1000 / (frame_rate(sample_rate) * BITS_PER_CODEBOOK))


# ─── Available EnCodec models (via Meta's encodec package) ────────────────────
AVAILABLE_MODELS: dict[str, int] = {
    "EnCodec 24 kHz · mono":   24000,
    "EnCodec 48 kHz · stereo": 48000,
}

# ─── Model cache (load once, reuse across Gradio calls) ───────────────────────
_model_cache: dict[str, tuple] = {}

def get_model(display_name: str) -> tuple:
    """Return (model, sample_rate), loading & caching on first call."""
    if display_name not in _model_cache:
        sr = AVAILABLE_MODELS[display_name]
        if sr == 24000:
            model = EncodecModel.encodec_model_24khz()
        else:
            model = EncodecModel.encodec_model_48khz()
        model = model.to(DEVICE)
        model.eval()
        _model_cache[display_name] = (model, sr)
        log.info("Loaded model '%s'  sr=%d Hz", display_name, sr)
    return _model_cache[display_name]


# ─── Audio utilities ───────────────────────────────────────────────────────────

def prepare_waveform(wav: torch.Tensor, is_stereo_model: bool) -> torch.Tensor:
    """Enforce the channel count expected by the chosen model."""
    C = wav.shape[0]
    if is_stereo_model:
        if C == 1:
            wav = wav.repeat(2, 1)       # mono → fake stereo
        elif C > 2:
            wav = wav[:2]                # keep first two channels
    else:
        if C > 1:
            wav = wav.mean(0, keepdim=True)   # mix down to mono
    return wav

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(1, C, T) → numpy array in Gradio audio format (T,) or (T, C)."""
    x = tensor.squeeze(0).cpu().numpy()    # (C, T)
    return x.T if x.ndim == 2 else x       # Gradio expects (T, C) for stereo

def snr_db(ref: np.ndarray, rec: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB between two (possibly multi-channel) signals."""
    min_len = min(ref.size, rec.size)
    r, s = ref.ravel()[:min_len], rec.ravel()[:min_len]
    noise_pwr = np.mean((r - s) ** 2)
    if noise_pwr < 1e-12:
        return float("inf")
    return 10 * math.log10(np.mean(r ** 2) / noise_pwr)

def encode_decode(model, wav_batch: torch.Tensor, bw: float) -> torch.Tensor | None:
    """Run one encode → decode cycle at the given bandwidth.  Returns None if unsupported."""
    try:
        model.set_target_bandwidth(bw)
        with torch.no_grad():
            encoded = model.encode(wav_batch)
            decoded = model.decode(encoded)
        return decoded
    except Exception as exc:
        log.warning("Bandwidth %.1f kbps not supported: %s", bw, exc)
        return None


# ─── Spectrogram helper ────────────────────────────────────────────────────────

def spectrogram_figure(signals: dict[str, np.ndarray], sr: int) -> plt.Figure:
    """
    Side-by-side spectrograms for all signals.
    `signals` maps label → 1-D numpy array (mono).
    """
    n = len(signals)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(3.5 * n, 7), 3),
        constrained_layout=True,
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    for ax, (label, sig) in zip(axes, signals.items()):
        ax.specgram(
            sig.ravel(),
            Fs=sr,
            NFFT=512,
            noverlap=384,
            cmap="magma",
            scale="dB",
        )
        ax.set_title(label, fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel("Frequency (Hz)", fontsize=7)
    fig.suptitle("Spectrogram comparison — original vs. RVQ reconstruction", fontsize=9)
    return fig


# ─── Main Gradio callback ──────────────────────────────────────────────────────

def run(audio_path: str | None, model_name: str, bw: float):
    """
    Gradio function.

    Returns
    -------
    [original_audio, recon_audio, spectrogram_fig, info_text]
    """
    if audio_path is None:
        return None, None, None, "Upload an audio file to start."

    model, sr = get_model(model_name)
    is_stereo  = "stereo" in model_name.lower()

    # ── Load & pre-process ────────────────────────────────────────────────────
    wav, orig_sr = torchaudio.load(audio_path, backend="soundfile")
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    wav = prepare_waveform(wav, is_stereo)

    wav_batch = wav.unsqueeze(0).to(DEVICE)     # (1, C, T)
    orig_np   = to_numpy(wav)                   # (T,) or (T, C)
    orig_mono = orig_np.ravel() if orig_np.ndim == 1 else orig_np[:, 0]

    # ── Encode → decode at selected bandwidth ────────────────────────────────
    n_cb    = n_codebooks(bw, sr)
    label   = f"{bw:.1f} kbps · {n_cb} CB"
    decoded = encode_decode(model, wav_batch, bw)

    header = (
        f"Model       : {model_name}\n"
        f"Sample rate : {sr} Hz  (frame rate {frame_rate(sr):.0f} Hz)\n"
        f"Duration    : {wav.shape[-1] / sr:.2f} s\n"
        f"Channels    : {wav.shape[0]}\n"
        f"Bandwidth   : {bw} kbps  ({n_cb} codebooks)\n"
    )

    if decoded is not None:
        rec_np   = to_numpy(decoded)
        rec_mono = rec_np.ravel() if rec_np.ndim == 1 else rec_np[:, 0]
        snr      = snr_db(orig_mono, rec_mono)
        info_text = header + f"SNR         : {snr:+.1f} dB"
        fig = spectrogram_figure({"Original": orig_mono, label: rec_mono}, sr)
        return (sr, orig_np), (sr, rec_np), fig, info_text
    else:
        info_text = header + "Status      : ✗ bandwidth not supported by this model"
        fig = spectrogram_figure({"Original": orig_mono}, sr)
        return (sr, orig_np), None, fig, info_text


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    model_choices = list(AVAILABLE_MODELS.keys())

    with gr.Blocks(title="EnCodec Demo", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
# EnCodec — RVQ Perceptual Codec Demo

Upload audio, choose a **bitrate**, and compare the original against the
RVQ reconstruction from Meta's **EnCodec** model.
        """)

        with gr.Row():
            # ── Controls ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                model_dd = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Model",
                )
                bw_radio = gr.Radio(
                    choices=ALL_BANDWIDTHS,
                    value=6.0,
                    label="Bitrate (kbps)",
                )
                audio_in = gr.Audio(
                    label="Upload Audio  (WAV / MP3 / FLAC …)",
                    type="filepath",
                )
                run_btn = gr.Button("Encode → Decode", variant="primary")
                info_box = gr.Textbox(
                    label="Run Info & SNR",
                    lines=8,
                    interactive=False,
                )

            # ── Audio outputs ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                orig_out = gr.Audio(
                    label="Original  (resampled to model SR)",
                    interactive=False,
                )
                recon_out = gr.Audio(
                    label="Reconstruction",
                    interactive=False,
                )

        # ── Spectrogram ───────────────────────────────────────────────────────
        gr.Markdown("### Spectrogram comparison")
        gr.Markdown(
            "_Artefacts appear as smearing, loss of high-frequency detail, or quantisation noise floors._"
        )
        spec_plot = gr.Plot(label="Spectrograms")

        # ── Wire up ───────────────────────────────────────────────────────────
        run_btn.click(
            fn=run,
            inputs=[audio_in, model_dd, bw_radio],
            outputs=[orig_out, recon_out, spec_plot, info_box],
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
