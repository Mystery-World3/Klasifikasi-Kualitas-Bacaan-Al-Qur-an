import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from src.config import Config
from src.model import ContrastiveModel # <--- Ganti Import
from src.utils import AudioUtil

INPUT_FOLDER = "data/unlabeled"       
OUTPUT_CSV = "laporan_hasil_prediksi.csv"
MODEL_PATH = "models/final_model_skripsi.pth" 
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # <--- PERUBAHAN DI SINI
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model Contrastive dimuat di {device}")
        return model, device
    else:
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        return None, None

def predict_single_file(file_path, model, device):
    try:
        input_tensor = AudioUtil.preprocess(file_path, add_noise=False)
        if input_tensor is None:
            raise ValueError("Gagal preprocess")

        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            max_prob, predicted_idx = torch.max(probs, 1)
            
        class_name = LABELS[predicted_idx.item()]
        
        probs_numpy = probs.cpu().numpy()[0]
        detail_probs = {label: f"{p*100:.2f}%" for label, p in zip(LABELS, probs_numpy)}
        
        return {
            "Filename": os.path.basename(file_path),
            "Prediksi": class_name,
            "Confidence": f"{max_prob.item()*100:.2f}%",
            **detail_probs
        }
    except Exception as e:
        return {
            "Filename": os.path.basename(file_path),
            "Prediksi": "ERROR",
            "Confidence": "0%",
            "Error Log": str(e)
        }

if __name__ == "__main__":
    model, device = load_trained_model()
    
    if model:
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.wav"))
        print(f"📂 Memproses {len(files)} file...")
        
        results = []
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)}...", end="\r")
            res = predict_single_file(f, model, device)
            results.append(res)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n✅ Selesai! Cek file: {OUTPUT_CSV}")  