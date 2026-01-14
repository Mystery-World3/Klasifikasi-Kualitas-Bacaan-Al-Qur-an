import streamlit as st
import os
import torch
import torch.nn.functional as F
import numpy as np
from src.config import Config
from src.model import ClassifierModel
from src.utils import AudioUtil

# KONFIGURASI
MODEL_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

# Pesan Detail untuk User
PESAN = {
    "Mumtaz": "🌟 Luar Biasa! Makhorijul huruf dan tajwid sangat sempurna.",
    "Jayyid Jiddan": "✅ Sangat Baik. Bacaan lancar dengan kesalahan yang sangat minim.",
    "Jayyid": "👍 Baik. Sudah memenuhi standar, namun perhatikan detail tajwid.",
    "Maqbul": "🆗 Cukup. Bacaan dapat dimengerti, namun perlu latihan rutin.",
    "Rosib": "⚠️ Kurang. Disarankan belajar intensif dengan pembimbing tahsin."
}

@st.cache_resource
def load_model():
    """Load model dan cache agar tidak reload setiap kali ada interaksi."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inisialisasi arsitektur model (ResNet18 Modified)
    model = ClassifierModel(num_classes=len(LABELS))
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load weights. map_location penting jika deploy di CPU (Streamlit Cloud)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"Gagal memuat bobot model: {e}")
            return None, None
    else:
        st.error(f"File model tidak ditemukan di: {MODEL_PATH}")
        return None, None

def predict_audio(file_path, model, device):
    # 1. Preprocessing (Audio -> Mel Spectrogram Tensor)
    input_tensor = AudioUtil.preprocess(file_path)
    
    if input_tensor is None:
        return None, None

    # 2. Tambah dimensi Batch (1, 1, Freq, Time) dan pindah ke Device
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. Inferensi
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        max_prob, predicted_idx = torch.max(probs, 1)
        
    class_name = LABELS[predicted_idx.item()]
    confidence = max_prob.item() * 100
    
    return class_name, confidence

# --- TAMPILAN WEB (STREAMLIT) ---
st.set_page_config(page_title="Tahsin AI", page_icon="🎙️")

st.title("🎙️ Analisis Kualitas Bacaan Al-Qur'an")
st.markdown("Sistem Cerdas berbasis **Deep Learning (ResNet-18)** untuk menilai kualitas tilawah.")

# Load Model
model, device = load_model()

# File Uploader
uploaded_file = st.file_uploader("Upload rekaman suara (.wav)", type=["wav"])

if uploaded_file and model:
    # Simpan file sementara
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Tampilkan Audio Player
    st.audio(temp_filename, format='audio/wav')
    
    if st.button("🔍 Analisis Sekarang"):
        with st.spinner('Sedang mengekstrak fitur Mel-Spectrogram & Melakukan Klasifikasi...'):
            prediksi, akurasi = predict_audio(temp_filename, model, device)
            
            if prediksi:
                # Tampilkan Hasil
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(label="Prediksi Kualitas", value=prediksi)
                
                with col2:
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{akurasi:.2f}%")
                
                # Kotak Pesan
                if prediksi == "Mumtaz":
                    st.success(PESAN[prediksi])
                elif prediksi in ["Jayyid Jiddan", "Jayyid"]:
                    st.info(PESAN[prediksi])
                else:
                    st.warning(PESAN[prediksi])
            else:
                st.error("Gagal memproses audio. Pastikan format file benar.")
            
            # Hapus file temp
            if os.path.exists(temp_filename):
                os.remove(temp_filename)