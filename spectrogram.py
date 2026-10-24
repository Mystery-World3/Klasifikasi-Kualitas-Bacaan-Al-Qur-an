"""Generate a privacy-safe Log-Mel Spectrogram figure from a random labeled WAV.

The selected audio filename is deliberately not printed, plotted, or used as
the output filename so that personal names in dataset filenames are not exposed.
"""

import glob
import os
import random

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config


def main():
    wav_files = glob.glob(
        os.path.join(Config.LABELED_DIR, "**", "*.wav"),
        recursive=True,
    )

    if not wav_files:
        raise FileNotFoundError(
            "Tidak ada file WAV di folder dataset berlabel. "
            "Periksa Config.LABELED_DIR terlebih dahulu."
        )

    # Randomly select an audio sample, but never reveal its path or filename.
    sample_path = random.choice(wav_files)
    audio, sample_rate = librosa.load(
        sample_path,
        sr=Config.SAMPLE_RATE,
        mono=True,
    )

    # Match the model's fixed 3-second input duration.
    audio = audio[: Config.N_SAMPLES]
    if len(audio) < Config.N_SAMPLES:
        audio = np.pad(
            audio,
            (0, Config.N_SAMPLES - len(audio)),
            mode="constant",
        )

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmin=Config.F_MIN,
        fmax=Config.F_MAX,
        power=2.0,
    )
    # Keep dB values for visualization; do not min-max normalize or resize here.
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=Config.HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=Config.F_MIN,
        fmax=Config.F_MAX,
        cmap="magma",
        vmin=-80,
        vmax=0,
        ax=ax,
    )

    ax.set_title("Log-Mel Spectrogram")
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Frekuensi Mel (Hz)")
    fig.colorbar(image, ax=ax, format="%+2.0f dB", label="Level relatif (dB)")
    fig.tight_layout()

    os.makedirs(Config.FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(Config.FIGURE_DIR, "log_mel_spectrogram.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print only the generic output path, never the selected source filename.
    print(f"Visualisasi Log-Mel Spectrogram tersimpan di: {save_path}")


if __name__ == "__main__":
    main()
