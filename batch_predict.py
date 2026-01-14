import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from src.config import Config
from src.model import ClassifierModel
from src.utils import AudioUtil

# ==========================================
# 1. cCONFIGURATION
# ==========================================
INPUT_FOLDER = "data/unlabeled/Nasser_Alqutami"       
OUTPUT_CSV = "laporan_hasil_prediksi.csv"  
MODEL_PATH = "models/final_model_skripsi.pth" 

# Mapping Label 
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']

# ==========================================
# 2. LOAD MODEL & PREDICTION FUNCTION
# ==========================================
def load_trained_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    
    # Model initialization
    model = ClassifierModel(num_classes=len(LABELS))
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Evaluation mode
        return model, device
    else:
        print(f"Error: File model tidak ditemukan di {model_path}")
        return None, None

def predict_single_file(file_path, model, device):
    """
    Fungsi ini melakukan prediksi untuk satu file.
    Menggunakan logika yang sama persis dengan predict.py Anda.
    """
    try:
        # --- 1. Preprocessing (use src.utils) ---
        sig = AudioUtil.open_audio(file_path, Config.SAMPLE_RATE)
        sig = AudioUtil.rechannel(sig)
        sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
        
        # Convert to Tensor [1, Channel, Time]
        input_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)
        
        # --- 2. Inference ---
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            max_prob, predicted_idx = torch.max(probs, 1)
            
        class_name = LABELS[predicted_idx.item()]
        confidence = max_prob.item() * 100
        all_probs = probs[0] # Probability tensor of all classes
        
        probs_detail_list = []
        for i, label in enumerate(LABELS):
            pct = all_probs[i].item() * 100
            probs_detail_list.append(f"{label}: {pct:.2f}%")
        
        probs_str = " | ".join(probs_detail_list)

        return {
            "Filename": os.path.basename(file_path),
            "Prediksi": class_name,
            "Confidence": f"{confidence:.2f}%",
            "Detail_Probabilitas": probs_str,
            "Status": "Sukses"
        }

    except Exception as e:
        return {
            "Filename": os.path.basename(file_path),
            "Prediksi": "ERROR",
            "Confidence": "0%",
            "Detail_Probabilitas": str(e),
            "Status": "Gagal"
        }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- MEMULAI BATCH PREDICTION ---")
    
    # 1. Load Model
    model, device = load_trained_model(MODEL_PATH)
    if not model:
        exit()

    # 2. Search all files .wav
    # Search for .wav files in the INPUT_FOLDER folder
    search_path = os.path.join(INPUT_FOLDER, "*.wav")
    audio_files = glob.glob(search_path)
    
    total_files = len(audio_files)
    print(f"\nDitemukan {total_files} file audio di folder '{INPUT_FOLDER}'")
    
    if total_files == 0:
        print("Tidak ada file .wav yang ditemukan. Cek folder data/unlabeled Anda.")
        exit()

    results = []

    # 3. Prediction Loop
    print("-" * 50)
    for i, file_path in enumerate(audio_files):
        print(f"[{i+1}/{total_files}] Memproses: {os.path.basename(file_path)} ...", end=" ")
        
        # Make predictions
        data = predict_single_file(file_path, model, device)
        results.append(data)
        
        # Print short status on the same line
        if data["Status"] == "Sukses":
            print(f"-> {data['Prediksi']} ({data['Confidence']})")
        else:
            print(f"-> ERROR")

    # 4. save to CSV
    print("-" * 50)
    if results:
        df = pd.DataFrame(results)
        
        # save
        df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\n✅ SELESAI! Hasil prediksi telah disimpan ke file:")
        print(f"   >> {os.path.abspath(OUTPUT_CSV)}")
        
        # Show trailer
        print("\nCuplikan 5 data pertama:")
        print(df[['Filename', 'Prediksi', 'Confidence']].head())
    else:
        print("Gagal memproses data.")