import streamlit as st
import os
import torch
import torch.nn.functional as F
from src.config import Config
from src.model import ClassifierModel
from src.utils import AudioUtil

# ==========================================
# 1. CONFIGURATION & LOAD MODEL
# ==========================================
MODEL_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

PESAN_PENJELASAN = {
    "Mumtaz": "Luar Biasa! Bacaan sangat fasih dan tajwid sempurna.",
    "Jayyid Jiddan": "Sangat Baik. Bacaan lancar, hanya ada kesalahan sangat kecil.",
    "Jayyid": "Baik. Sudah bagus, namun perlu perbaikan di beberapa tempat.",
    "Maqbul": "Cukup. Bacaan masih bisa dimengerti, tapi butuh banyak latihan.",
    "Rosib": "Kurang. Perlu belajar tahsin lebih intensif lagi."
}

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassifierModel(num_classes=len(LABELS))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    return None, None

def predict_audio_file(file_path, model, device):
    # Preprocessing
    sig = AudioUtil.open_audio(file_path, Config.SAMPLE_RATE)
    sig = AudioUtil.rechannel(sig)
    sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
    input_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        max_prob, predicted_idx = torch.max(probs, 1)
        
    hasil = LABELS[predicted_idx.item()]
    return hasil

# ==========================================
# 2. USER INTERTACE (UI)
# ==========================================
st.title("🎙️ Sistem Penilai Tahsin Otomatis")
st.write("Upload rekaman suara Anda, dan AI akan menilai kualitas bacaannya.")

# Load Model at the beginning
model, device = load_model()

if model is None:
    st.error("Model tidak ditemukan! Pastikan file model ada di folder 'models/'.")
else:
    # File Upload Component
    uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        # Show Audio Player so user can listen
        st.audio(uploaded_file, format='audio/wav')

        if st.button("🔍 Nilai Bacaan Saya"):
            with st.spinner('Sedang mendengarkan dan menganalisis...'):
                # Save a temporary file so that it can be read by the AudioUtil library.
                temp_filename = "temp_audio.wav"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Prediction Process
                hasil_prediksi = predict_audio_file(temp_filename, model, device)
                
                # SHOW RESULTS
                st.markdown("---")
                st.subheader("Hasil Penilaian:")
                
                # Color logic
                if hasil_prediksi == "Mumtaz":
                    st.success(f"🌟 **{hasil_prediksi}**")
                elif hasil_prediksi in ["Jayyid Jiddan", "Jayyid"]:
                    st.info(f"✅ **{hasil_prediksi}**")
                else:
                    st.warning(f"⚠️ **{hasil_prediksi}**")
                
                # displays an explanatory message
                st.write(f"📝 *Keterangan: {PESAN_PENJELASAN[hasil_prediksi]}*")
                
                # delete temporary files
                os.remove(temp_filename)