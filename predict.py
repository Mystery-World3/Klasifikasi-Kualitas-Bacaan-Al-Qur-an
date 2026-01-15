import argparse
import torch
import torch.nn.functional as F
from src.model import ContrastiveModel  # <--- Ganti Import
from src.utils import AudioUtil

MODEL_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

def main(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # <--- PERUBAHAN DI SINI
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune')
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print("❌ Model belum dilatih! Jalankan train.py dulu.")
        return

    print(f"🔍 Menganalisis: {audio_path}")
    tensor = AudioUtil.preprocess(audio_path, add_noise=False)
    
    if tensor is not None:
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = F.softmax(output, dim=1)
            score, idx = torch.max(prob, 1)
            
        print("-" * 30)
        print(f"🎯 HASIL: {LABELS[idx.item()]}")
        print(f"📊 CONFIDENCE: {score.item()*100:.2f}%")
        print("-" * 30)
    else:
        print("❌ Gagal memproses file audio.")

if __name__ == "__main__":
    # Ganti path ini untuk tes cepat
    TEST_FILE = "data/unlabeled/1.wav" 
    main(TEST_FILE)