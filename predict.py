import os
import argparse
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
from collections import Counter

from src.config import Config
from src.model import ContrastiveModel
from src.utils import AudioUtil

MODEL_PATH = os.path.join(Config.MODEL_DIR, "classifier_seed_42.pth")

# Gunakan huruf kapital sesuai label yang dipakai waktu training
LABELS = ["MUMTAZ", "JAYYID_JIDDAN", "JAYYID", "MAQBUL", "RASIB"]

def main(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File audio '{audio_path}' tidak ditemukan!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune')
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model belum dilatih! File {MODEL_PATH} tidak ditemukan.")
        return

    print(f"\n🎙️ Menganalisis file: {audio_path}")
    print("Menerapkan teknik Sliding Window (potongan 3 detik)...")
    
    # 2. Load Audio Penuh
    try:
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
    except Exception as e:
        print(f"Gagal membaca file audio: {e}")
        return
        
    chunk_length = Config.N_SAMPLES # 22050 * 3 detik = 66150
    
    # Padding jika rekaman kurang dari 3 detik
    if len(audio) < chunk_length:
        pad_length = chunk_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
        
    # 3. Potong-potong audio menjadi chunk 3 detik
    chunks = []
    for i in range(0, len(audio) - chunk_length + 1, chunk_length):
        chunks.append(audio[i : i + chunk_length])
        
    # Ambil sisa di bagian akhir jika lumayan panjang
    if len(audio) % chunk_length > (chunk_length // 2):
        chunks.append(audio[-chunk_length:])
        
    predictions = []
    probs_list = []
    
    print("-" * 45)
    print("Rincian Analisis per 3 Detik:")
    
    # 4. Evaluasi setiap potongan (chunk)
    for idx, chunk in enumerate(chunks):
        # Simpan sementara agar AudioUtil bisa membaca filenya
        temp_chunk_path = f"temp_chunk_cli_{idx}.wav"
        sf.write(temp_chunk_path, chunk, sr)
        
        tensor = AudioUtil.preprocess(temp_chunk_path, add_noise=False)
        os.remove(temp_chunk_path) # Hapus file sementara
        
        if tensor is not None:
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                prob = F.softmax(output, dim=1)
                score, pred_idx = torch.max(prob, 1)
                
            label = LABELS[pred_idx.item()]
            confidence = score.item() * 100
            
            predictions.append(pred_idx.item())
            probs_list.append(confidence)
            
            print(f"⏱️ Detik {idx*3:02d}-{(idx+1)*3:02d} | Label: {label:<14} | Conf: {confidence:.2f}%")
            
    print("-" * 45)
    
    if not predictions:
        print("Gagal mengekstrak fitur dari audio.")
        return
        
    # 5. Penarikan Kesimpulan (Majority Vote)
    vote_counts = Counter(predictions)
    majority_idx = vote_counts.most_common(1)[0][0]
    
    # Hitung rata-rata confidence khusus dari vote yang menang
    majority_probs = [p for p, i in zip(probs_list, predictions) if i == majority_idx]
    final_confidence = sum(majority_probs) / len(majority_probs)
    
    print("\n" + "=" * 45)
    print("KESIMPULAN AKHIR (MAJORITY VOTE)")
    print("=" * 45)
    print(f"KUALITAS TAHSIN : {LABELS[majority_idx]}")
    print(f"CONFIDENCE      : {final_confidence:.2f}%")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi Kualitas Tahsin dengan Sliding Window")
    parser.add_argument("--audio", type=str, default="data/unlabeled/1.wav", help="Path lengkap ke file rekaman .wav yang belum dipotong")
    
    args = parser.parse_args()
    main(args.audio)