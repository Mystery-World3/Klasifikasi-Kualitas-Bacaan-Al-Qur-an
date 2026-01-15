import streamlit as st
import os
import torch
import torch.nn.functional as F
from src.config import Config
from src.model import ContrastiveModel  
from src.utils import AudioUtil

# KONFIGURASI
MODEL_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

PESAN = {
    "Mumtaz": "🌟 Luar Biasa! Makhorijul huruf dan tajwid sangat sempurna.",
    "Jayyid Jiddan": "✅ Sangat Baik. Bacaan lancar dengan kesalahan yang sangat minim.",
    "Jayyid": "👍 Baik. Sudah memenuhi standar, namun perhatikan detail tajwid.",
    "Maqbul": "🆗 Cukup. Bacaan dapat dimengerti, namun perlu latihan rutin.",
    "Rosib": "⚠️ Kurang. Disarankan belajar intensif dengan pembimbing tahsin."
}

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune')
    
    if os.path.exists(MODEL_PATH):
        try:
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
    # Tidak perlu add_noise saat prediksi
    input_tensor = AudioUtil.preprocess(file_path, add_noise=False)
    
    if input_tensor is None:
        return None, None

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor) # Output langsung logits karena mode='finetune'
        probs = F.softmax(logits, dim=1)
        max_prob, predicted_idx = torch.max(probs, 1)
        
    class_name = LABELS[predicted_idx.item()]
    confidence = max_prob.item() * 100
    
    return class_name, confidence

# --- TAMPILAN WEB ---
st.set_page_config(page_title="Tahsin AI", page_icon="🎙️")
st.title("🎙️ Analisis Kualitas Bacaan Al-Qur'an (Contrastive AI)")
st.markdown("Sistem ini menggunakan **Semi-Supervised Contrastive Learning** untuk hasil yang lebih akurat.")

model, device = load_model()
uploaded_file = st.file_uploader("Upload rekaman suara (.wav)", type=["wav"])

if uploaded_file and model:
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(temp_filename, format='audio/wav')
    
    if st.button("🔍 Analisis Sekarang"):
        with st.spinner('Sedang menganalisis fitur audio...'):
            prediksi, akurasi = predict_audio(temp_filename, model, device)
            
            if prediksi:
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Kualitas", prediksi)
                with col2:
                    st.metric("Confidence", f"{akurasi:.2f}%")
                
                if prediksi == "Mumtaz": st.success(PESAN[prediksi])
                elif prediksi in ["Jayyid Jiddan", "Jayyid"]: st.info(PESAN[prediksi])
                else: st.warning(PESAN[prediksi])
            else:
                st.error("Gagal memproses audio.")
            
            if os.path.exists(temp_filename): os.remove(temp_filename)