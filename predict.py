import argparse
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.config import Config
from src.model import ContrastiveModel


# CONFIGURATION

HF_REPO_ID = "Mysteryworld3/quran-recitation-classifier"
MODEL_FILENAME = "classifier_seed_82.pth"

LABELS = [
    "MUMTAZ",
    "JAYYID_JIDDAN",
    "JAYYID",
    "MAQBUL",
    "RASIB",
]


# AUDIO PREPROCESSING

def audio_to_tensor(audio):
    """
    Mengubah audio 3 detik menjadi tensor input model.

    Pipeline:
    audio
      -> pad / truncate
      -> Mel Spectrogram
      -> power_to_db (Log Mel)
      -> normalisasi 0-1
      -> resize 64x64
      -> tensor 1-channel
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Pastikan panjang 3 detik

    audio = audio[:Config.N_SAMPLES]

    if len(audio) < Config.N_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                Config.N_SAMPLES - len(audio)
            ),
            mode="constant"
        )

    # Mel Spectrogram

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )

    # Log Mel Spectrogram

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalisasi

    mel = (
        mel - mel.min()
    ) / (
        mel.max() - mel.min() + 1e-8
    )

    # Convert ke Tensor

    tensor = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0)

    # Resize 64 x 64

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return tensor


# SPLIT AUDIO

def split_audio(audio):
    """
    Membagi audio menjadi potongan 3 detik.

    Sisa audio hanya digunakan jika lebih dari
    setengah durasi chunk.
    """

    chunk_length = Config.N_SAMPLES

    chunks = []

    # Audio <= 3 detik

    if len(audio) <= chunk_length:

        chunk = np.pad(
            audio,
            (
                0,
                max(
                    0,
                    chunk_length - len(audio)
                )
            ),
            mode="constant"
        )

        chunks.append(
            chunk
        )

        return chunks

    # Potongan utama

    for start in range(
        0,
        len(audio) - chunk_length + 1,
        chunk_length
    ):

        chunks.append(
            audio[
                start:start + chunk_length
            ]
        )

    # Sisa audio

    remainder = len(audio) % chunk_length

    if remainder > chunk_length // 2:

        chunks.append(
            audio[-chunk_length:]
        )

    return chunks


# DOWNLOAD MODEL

def load_model():
    """
    Download checkpoint dari Hugging Face dan
    memuat model untuk inference.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("TAHSIN AI - MODEL LOADER")
    print("=" * 60)

    print(
        f"Repository : {HF_REPO_ID}"
    )

    print(
        f"Checkpoint : {MODEL_FILENAME}"
    )

    print(
        f"Device     : {device}"
    )

    print()
    print(
        "Mengambil model dari Hugging Face..."
    )

    try:

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
        )

    except Exception as e:

        print()
        print(
            "ERROR: Gagal mengambil model dari Hugging Face."
        )

        print(
            f"Detail: {e}"
        )

        raise SystemExit(1)

    print(
        f"Model path : {model_path}"
    )

    print(
        "Membangun arsitektur ResNet-18..."
    )

    try:

        model = ContrastiveModel(
            num_classes=len(LABELS),
            mode="finetune",
            pretrained=False,
        )

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint
        )

        model.to(device)
        model.eval()

    except Exception as e:

        print()
        print(
            "ERROR: Gagal memuat checkpoint."
        )

        print(
            f"Detail: {e}"
        )

        raise SystemExit(1)

    print(
        "Model berhasil dimuat."
    )

    print("=" * 60)

    return model, device


# PREDICTION

def predict_audio(
    audio_path,
    model,
    device,
):
    """
    Melakukan inference terhadap seluruh file audio.
    """

    # Validasi file

    if not os.path.exists(audio_path):

        print(
            f"ERROR: File tidak ditemukan: {audio_path}"
        )

        return None

    # Load audio

    print()
    print(
        f"Membaca audio: {audio_path}"
    )

    try:

        audio, sr = librosa.load(
            audio_path,
            sr=Config.SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        print(
            f"ERROR: Gagal membaca audio: {e}"
        )

        return None

    duration = len(audio) / sr

    print(
        f"Sample rate : {sr} Hz"
    )

    print(
        f"Durasi      : {duration:.2f} detik"
    )

    # Split

    chunks = split_audio(
        audio
    )

    print(
        f"Jumlah chunk: {len(chunks)}"
    )

    print()

    # Prediction

    all_probabilities = []

    print("-" * 75)

    print(
        f"{'Chunk':<8}"
        f"{'Waktu':<18}"
        f"{'Prediksi':<20}"
        f"{'Confidence':>12}"
    )

    print("-" * 75)

    for index, chunk in enumerate(chunks):

        # Feature extraction

        tensor = audio_to_tensor(
            chunk
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(device)
        )

        # Inference

        with torch.no_grad():

            logits = model(
                tensor
            )

            probs = F.softmax(
                logits,
                dim=1
            )

        probs_numpy = (
            probs
            .squeeze(0)
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            probs_numpy
        )

        # Chunk prediction

        pred_index = int(
            np.argmax(
                probs_numpy
            )
        )

        label = LABELS[
            pred_index
        ]

        confidence = (
            probs_numpy[
                pred_index
            ] * 100
        )

        start_time = index * 3
        end_time = (index + 1) * 3

        time_range = (
            f"{start_time:02d}"
            f"-"
            f"{end_time:02d} detik"
        )

        print(
            f"{index + 1:<8}"
            f"{time_range:<18}"
            f"{label:<20}"
            f"{confidence:>10.2f}%"
        )

    print("-" * 75)

    # Tidak ada hasil

    if len(all_probabilities) == 0:

        print(
            "ERROR: Tidak ada hasil prediction."
        )

        return None

    # FINAL AGGREGATION

    probability_matrix = np.array(
        all_probabilities
    )

    # Rata-rata probabilitas seluruh chunk
    average_probabilities = (
        probability_matrix.mean(
            axis=0
        )
    )

    # Final class
    final_index = int(
        np.argmax(
            average_probabilities
        )
    )

    final_label = LABELS[
        final_index
    ]

    final_confidence = (
        average_probabilities[
            final_index
        ] * 100
    )

    # FINAL RESULT

    print()
    print("=" * 60)
    print("KESIMPULAN AKHIR")
    print("=" * 60)

    print(
        f"KUALITAS TAHSIN : {final_label}"
    )

    print(
        f"CONFIDENCE      : {final_confidence:.2f}%"
    )

    print("=" * 60)

    # ALL CLASS PROBABILITIES

    print()
    print(
        "PROBABILITAS RATA-RATA SETIAP KELAS"
    )

    print("-" * 40)

    for label, probability in zip(
        LABELS,
        average_probabilities
    ):

        print(
            f"{label:<20}"
            f"{probability * 100:>8.2f}%"
        )

    print("-" * 40)

    return {
        "label": final_label,
        "confidence": final_confidence,
        "probabilities": average_probabilities,
    }


# MAIN

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prediksi kualitas bacaan Al-Qur'an "
            "menggunakan Tahsin AI."
        )
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help=(
            "Path file audio WAV yang ingin dianalisis."
        ),
    )

    args = parser.parse_args()

    # Load model

    model, device = load_model()

    # Prediction

    predict_audio(
        audio_path=args.audio,
        model=model,
        device=device,
    )


# ENTRY POINT

if __name__ == "__main__":
    main()