import streamlit as st
import os
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
from collections import Counter
from src.config import Config
from src.model import ContrastiveModel  
from src.utils import AudioUtil

MODEL_PATH = os.path.join(
    Config.MODEL_DIR,
    "classifier_seed_42.pth"
)
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

PESAN = {
    "Mumtaz": "Luar Biasa! Makhorijul huruf dan tajwid sangat sempurna.",
    "Jayyid Jiddan": "Sangat Baik. Bacaan lancar dengan kesalahan yang sangat minim.",
    "Jayyid": "Baik. Sudah memenuhi standar, namun perhatikan detail tajwid.",
    "Maqbul": "Cukup. Bacaan dapat dimengerti, namun perlu latihan rutin.",
    "Rosib": "Kurang. Disarankan belajar intensif dengan pembimbing tahsin."
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

def predict_audio_sliding_window(file_path, model, device):
    # 1. Load audio penuh
    audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
    chunk_length = Config.N_SAMPLES # 22050 * 3 detik = 66150
    
    if len(audio) < chunk_length:
        pad_length = chunk_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
        
    chunks = []
    for i in range(0, len(audio) - chunk_length + 1, chunk_length):
        chunks.append(audio[i : i + chunk_length])
        
    if len(audio) % chunk_length > (chunk_length // 2):
        chunks.append(audio[-chunk_length:])
        
    predictions = []
    probs_list = []
    
    for idx, chunk in enumerate(chunks):
        temp_chunk_path = f"temp_chunk_{idx}.wav"
        sf.write(temp_chunk_path, chunk, sr)
        
        input_tensor = AudioUtil.preprocess(temp_chunk_path, add_noise=False)
        os.remove(temp_chunk_path) # Bersihkan file sementara
        
        if input_tensor is not None:
            input_tensor = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor) 
                probs = F.softmax(logits, dim=1)
                max_prob, predicted_idx = torch.max(probs, 1)
                
            predictions.append(predicted_idx.item())
            probs_list.append(max_prob.item())
            
    if not predictions:
        return None, None, None

    vote_counts = Counter(predictions)
    majority_idx = vote_counts.most_common(1)[0][0]
    
    majority_probs = [p for p, i in zip(probs_list, predictions) if i == majority_idx]
    confidence = (sum(majority_probs) / len(majority_probs)) * 100
    
    class_name = LABELS[majority_idx]
    
    detail_potongan = [(LABELS[p], conf * 100) for p, conf in zip(predictions, probs_list)]
    
    return class_name, confidence, detail_potongan

st.set_page_config(page_title="Tahsin AI", page_icon="🎙️")
st.title("Analisis Kualitas Bacaan Al-Qur'an (Contrastive AI)")
st.markdown("Sistem ini memecah rekaman panjang menjadi potongan 3 detik dan mengambil rata-rata kualitas keseluruhannya.")

model, device = load_model()
uploaded_file = st.file_uploader("Upload rekaman suara (.wav)", type=["wav"])

if uploaded_file and model:
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(temp_filename, format='audio/wav')
    
    if st.button("Analisis Sekarang"):
        with st.spinner('Sedang menganalisis fitur audio...'):
            prediksi, akurasi, rincian = predict_audio_sliding_window(temp_filename, model, device)
            
            if prediksi:
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Kualitas Keseluruhan", prediksi)
                with col2:
                    st.metric("Tingkat Keyakinan (Confidence)", f"{akurasi:.2f}%")
                
                if prediksi == "Mumtaz": st.success(PESAN[prediksi])
                elif prediksi in ["Jayyid Jiddan", "Jayyid"]: st.info(PESAN[prediksi])
                else: st.warning(PESAN[prediksi])
                
                # Tampilkan rincian performa tiap potongan detik
                with st.expander("Lihat Rincian Analisis per 3 Detik"):
                    for i, (label_potongan, conf_potongan) in enumerate(rincian):
                        st.write(f"Detik ke-{i*3} - {(i+1)*3} : **{label_potongan}** ({conf_potongan:.1f}%)")
            else:
                st.error("Gagal memproses audio.")
            
            if os.path.exists(temp_filename): os.remove(temp_filename)