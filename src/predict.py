# predict.py
import torch
import torch.nn.functional as F
import os
from src.config import Config
from src.model import ClassifierModel
from src.utils import AudioUtil

# Mapping Label (Harus sama urutannya dengan saat training)
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

def load_trained_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inisialisasi model kosong dengan struktur yang sama
    model = ClassifierModel(num_classes=5)
    
    # Load nyawa (weights) yang sudah dilatih
    if os.path.exists(model_path):
        # map_location='cpu' penting jika laptop tidak ada GPU saat tes
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Mode evaluasi (matikan dropout, dll)
        return model, device
    else:
        print(f"Error: File model tidak ditemukan di {model_path}")
        return None, None

def predict_audio(file_path, model, device):
    # 1. Preprocessing (Sama persis dengan saat training)
    # Load audio
    sig = AudioUtil.open_audio(file_path, Config.SAMPLE_RATE)
    sig = AudioUtil.rechannel(sig)
    sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
    
    # Ubah ke Tensor dan tambah dimensi Batch [1, Channel, Time]
    input_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Prediksi
    with torch.no_grad(): # Hemat memori
        logits = model(input_tensor)
        
        # Ubah skor aneh (logits) menjadi persen (0-1)
        probs = F.softmax(logits, dim=1)
        
        # Ambil nilai tertinggi
        max_prob, predicted_idx = torch.max(probs, 1)
        
    class_name = LABELS[predicted_idx.item()]
    confidence = max_prob.item() * 100
    
    return class_name, confidence, probs[0]

if __name__ == "__main__":

    TEST_FILE = "data/unlabeled/Yasser_001.wav" 
    
    MODEL_PATH = "models/final_model_skripsi.pth"
    
    print("--- MULAI PREDIKSI ---")
    
 
    if not os.path.exists(TEST_FILE):
        print(f"ERROR FATAL: File suara '{TEST_FILE}' tidak ditemukan!")
        print("Pastikan Anda menaruh file .wav di folder data/unlabeled dan namanya sesuai.")
        exit()

    # Load Model
    model, device = load_trained_model(MODEL_PATH)
    
    if model:
        print(f"Sedang mendengarkan file: {TEST_FILE} ...")
        result, score, all_probs = predict_audio(TEST_FILE, model, device)
        
        print("\n" + "="*30)
        print(f"HASIL PREDIKSI AI")
        print("="*30)
        print(f"Kualitas  : \033[92m{result}\033[0m") # Teks Hijau
        print(f"Keyakinan : {score:.2f}%")
        print("-" * 30)
        
        print("Rincian Probabilitas:")
        for i, label in enumerate(LABELS):
            persen = all_probs[i].item() * 100
            print(f"- {label:<15} : {persen:.2f}%")