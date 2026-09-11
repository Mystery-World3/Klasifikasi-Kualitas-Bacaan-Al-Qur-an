
import io

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download

from src.config import Config
from src.model import ContrastiveModel


# KONFIGURASI
HF_REPO_ID = "Mysteryworld3/quran-recitation-classifier"
MODEL_FILENAME = "classifier_seed_82.pth"

LABELS = [
    "Mumtaz",
    "Jayyid Jiddan",
    "Jayyid",
    "Maqbul",
    "Rasib"
]

PESAN = {
    "Mumtaz": (
        "Luar biasa! Bacaan menunjukkan kualitas yang sangat baik "
    ),
    "Jayyid Jiddan": (
        "Sangat baik. Bacaan sudah lancar dengan sedikit kesalahan "
    ),
    "Jayyid": (
        "Baik. Bacaan sudah memenuhi standar, namun masih perlu "
        "memperhatikan beberapa detail tajwid."
    ),
    "Maqbul": (
        "Cukup. Bacaan masih dapat dipahami, namun diperlukan "
        "latihan tahsin secara rutin."
    ),
    "Rasib": (
        "Perlu peningkatan. Disarankan melakukan latihan tahsin "
        "secara lebih intensif dengan pembimbing."
    )
}


# PAGE CONFIG
st.set_page_config(
    page_title="Tahsin AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9aa0a6;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .small-text {
        font-size: 13px;
        color: #9aa0a6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# SIDEBAR
with st.sidebar:

    st.markdown("## Tahsin AI")

    st.markdown(
        """
        Sistem klasifikasi kualitas bacaan
        Al-Qur'an berbasis:

        **Semi-Supervised Contrastive Learning**

        - Log Mel Spectrogram
        - ResNet-18
        - 5 kelas kualitas bacaan
        """
    )

    st.markdown("---")

    st.markdown("### Model")

    st.write(
        "**Hugging Face:**"
    )

    st.code(
        HF_REPO_ID,
        language="text"
    )

    st.write(
        "**Checkpoint:**"
    )

    st.code(
        MODEL_FILENAME,
        language="text"
    )

    st.markdown("---")

    st.markdown("### Kelas")

    for label in LABELS:
        st.write(f"• {label}")

    st.markdown("---")

    st.caption(
        "Tahsin AI • Semi-Supervised Contrastive Learning"
    )


# HEADER
st.markdown(
    '<div class="main-title">Analisis Kualitas Bacaan Al-Qur\'an</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analisis otomatis kualitas bacaan menggunakan
    <b>Log Mel Spectrogram</b> dan model
    <b>ResNet-18</b> yang dilatih dengan pendekatan
    <b>Semi-Supervised Contrastive Learning</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# MODEL
@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Download checkpoint dari Hugging Face
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    # Buat arsitektur model
    # ImageNet weights
    model = ContrastiveModel(
        num_classes=len(LABELS),
        mode="finetune",
        pretrained=False
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    return model, device, model_path


# LOAD MODEL
try:

    with st.spinner(
        "Memuat model dari Hugging Face..."
    ):

        model, device, model_path = load_model()

    st.success(
        "Model berhasil dimuat dan siap digunakan."
    )

except Exception as e:

    st.error(
        "Gagal memuat model."
    )

    st.exception(e)

    st.stop()


# UPLOAD AUDIO
st.markdown("---")

st.markdown(
    '<div class="section-title">Upload Rekaman</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload rekaman bacaan Al-Qur'an dalam format WAV."
)

uploaded_file = st.file_uploader(
    "Pilih file audio",
    type=["wav"],
    label_visibility="collapsed"
)


# AUDIO PREPROCESSING
def audio_to_tensor(audio):

    """
    Mengubah audio menjadi input model.

    Pipeline:
    Audio
      ↓
    22050 Hz
      ↓
    3 detik
      ↓
    Mel Spectrogram
      ↓
    Power to dB
      ↓
    Normalisasi
      ↓
    Resize 64 x 64
      ↓
    Tensor 1-channel
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Potong
    audio = audio[:Config.N_SAMPLES]

    # Padding
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
        n_mels=Config.N_MELS
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

    # Tensor
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

        chunks.append(chunk)

        return chunks

    # Potongan utama
    for start in range(
        0,
        len(audio) - chunk_length + 1,
        chunk_length
    ):

        chunk = audio[
            start:start + chunk_length
        ]

        chunks.append(
            chunk
        )

    # Sisa audio
    remainder = len(audio) % chunk_length

    if remainder > chunk_length // 2:

        chunks.append(
            audio[-chunk_length:]
        )

    return chunks


# PREDICTION
def predict_audio(
    audio,
    model,
    device
):

    chunks = split_audio(
        audio
    )

    predictions = []
    probabilities = []
    spectrograms = []

    # PREDICT SETIAP CHUNK
    for chunk in chunks:

        # Preprocessing
        tensor = audio_to_tensor(
            chunk
        )

        # Simpan spectrogram
        spectrograms.append(
            tensor.squeeze(0).numpy()
        )

        # Tambahkan batch dimension
        tensor = tensor.unsqueeze(0).to(
            device
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

        # Probability
        probs_numpy = (
            probs
            .squeeze(0)
            .cpu()
            .numpy()
        )

        probabilities.append(
            probs_numpy
        )

        # Predicted class
        predicted_index = int(
            np.argmax(
                probs_numpy
            )
        )

        predictions.append(
            predicted_index
        )

    # AGGREGATE
    probability_matrix = np.array(
        probabilities
    )

    average_probabilities = (
        probability_matrix.mean(
            axis=0
        )
    )

    # Final prediction
    final_index = int(
        np.argmax(
            average_probabilities
        )
    )

    final_label = LABELS[
        final_index
    ]

    confidence = (
        average_probabilities[
            final_index
        ] * 100
    )

    # DETAIL PER CHUNK
    details = []

    for i, probs in enumerate(
        probability_matrix
    ):

        predicted_index = int(
            np.argmax(probs)
        )

        predicted_label = LABELS[
            predicted_index
        ]

        chunk_confidence = (
            probs[predicted_index] * 100
        )

        start_time = i * 3
        end_time = (i + 1) * 3

        details.append(
            {
                "Potongan": i + 1,
                "Waktu": (
                    f"{start_time:.0f}–"
                    f"{end_time:.0f} detik"
                ),
                "Prediksi": predicted_label,
                "Confidence": chunk_confidence
            }
        )

    return (
        final_label,
        confidence,
        average_probabilities,
        details,
        spectrograms
    )


# AUDIO PROCESS
if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    # Audio Player
    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    # Load audio
    try:

        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=Config.SAMPLE_RATE,
            mono=True
        )

    except Exception as e:

        st.error(
            f"Gagal membaca audio: {e}"
        )

        st.stop()

    # Audio information
    duration = (
        len(audio) / sr
    )

    chunks = split_audio(
        audio
    )

    st.markdown("### Informasi Audio")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Durasi",
            f"{duration:.2f} detik"
        )

    with col2:

        st.metric(
            "Sample Rate",
            f"{sr:,} Hz"
        )

    with col3:

        st.metric(
            "Potongan",
            len(chunks)
        )

    with col4:

        st.metric(
            "Durasi / Potongan",
            "3 detik"
        )

    st.markdown("")


    # ANALYZE BUTTON
    analyze = st.button(
        "Analisis Bacaan Sekarang",
        type="primary",
        use_container_width=True
    )

    if analyze:

        # Prediction
        with st.spinner(
            "AI sedang menganalisis bacaan..."
        ):

            (
                prediction,
                confidence,
                average_probabilities,
                details,
                spectrograms
            ) = predict_audio(
                audio,
                model,
                device
            )

        # RESULT
        st.markdown("---")

        st.markdown(
            '<div class="section-title">Hasil Analisis</div>',
            unsafe_allow_html=True
        )

        st.markdown("")


        # Main result
        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Kualitas Bacaan",
                prediction
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        # Recommendation
        if prediction == "Mumtaz":

            st.success(
                f"**{prediction}** — {PESAN[prediction]}"
            )

        elif prediction in [
            "Jayyid Jiddan",
            "Jayyid"
        ]:

            st.info(
                f"**{prediction}** — {PESAN[prediction]}"
            )

        else:

            st.warning(
                f"**{prediction}** — {PESAN[prediction]}"
            )


        # TABS
        tab1, tab2, tab3 = st.tabs(
            [
                "Probabilitas",
                "Per Potongan",
                "5Spectrogram"
            ]
        )


        # TAB 1 — PROBABILITY
        with tab1:

            st.markdown(
                "### Probabilitas Prediksi"
            )

            probability_percent = (
                average_probabilities * 100
            )

            # DataFrame
            probability_df = pd.DataFrame(
                {
                    "Kelas": LABELS,
                    "Probabilitas (%)": probability_percent
                }
            )

            probability_df[
                "Probabilitas (%)"
            ] = probability_df[
                "Probabilitas (%)"
            ].round(2)

            # Chart
            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            bars = ax.bar(
                LABELS,
                probability_percent
            )

            ax.set_title(
                "Probabilitas Kualitas Bacaan"
            )

            ax.set_ylabel(
                "Probabilitas (%)"
            )

            ax.set_ylim(
                0,
                max(
                    100,
                    float(
                        probability_percent.max()
                    ) + 10
                )
            )

            ax.tick_params(
                axis="x",
                rotation=20
            )

            # Nilai di atas bar
            for bar, value in zip(
                bars,
                probability_percent
            ):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height()
                    + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # Probability table
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


        # TAB 2 — PER CHUNK
        with tab2:

            st.markdown(
                "### Analisis Per Potongan Audio"
            )

            st.write(
                f"Audio dianalisis menjadi "
                f"**{len(details)} potongan** "
                f"dengan durasi sekitar 3 detik per potongan."
            )

            # DataFrame
            detail_df = pd.DataFrame(
                details
            )

            # Statistik prediksi
            prediction_counts = (
                detail_df[
                    "Prediksi"
                ]
                .value_counts()
                .reindex(
                    LABELS,
                    fill_value=0
                )
            )

            st.markdown(
                "#### Distribusi Prediksi"
            )

            distribution_df = pd.DataFrame(
                {
                    "Kelas": prediction_counts.index,
                    "Jumlah Potongan": (
                        prediction_counts.values
                    )
                }
            )

            st.dataframe(
                distribution_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "#### Detail"
            )

            display_df = detail_df.copy()

            display_df[
                "Confidence"
            ] = display_df[
                "Confidence"
            ].map(
                lambda x: f"{x:.2f}%"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=500
            )


        # TAB 3 — SPECTROGRAM
        with tab3:

            st.markdown(
                "### Log Mel Spectrogram"
            )

            st.write(
                "Visualisasi fitur audio yang digunakan "
                "sebagai input model."
            )

            # Select chunk
            if len(spectrograms) > 0:

                selected_chunk = st.selectbox(
                    "Pilih potongan audio",
                    options=list(
                        range(
                            len(spectrograms)
                        )
                    ),
                    format_func=lambda x:
                        f"Potongan {x + 1} "
                        f"({x * 3}–{(x + 1) * 3} detik)"
                )

                selected_spectrogram = (
                    spectrograms[
                        selected_chunk
                    ]
                )

                # Plot
                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                image = ax.imshow(
                    selected_spectrogram,
                    aspect="auto",
                    origin="lower"
                )

                ax.set_title(
                    "Log Mel Spectrogram"
                )

                ax.set_xlabel(
                    "Time"
                )

                ax.set_ylabel(
                    "Mel Frequency"
                )

                fig.colorbar(
                    image,
                    ax=ax,
                    label="Normalized dB"
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)


# NO AUDIO MESSAGE
else:

    st.info(
        "Silakan upload rekaman bacaan Al-Qur'an "
        "berformat WAV untuk memulai analisis."
    )


# FOOTER
st.markdown("---")

st.caption(
    "Tahsin AI • Semi-Supervised Contrastive Learning "
    "• Log Mel Spectrogram • ResNet-18"
)