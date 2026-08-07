# check_data.py
import os
import glob
import torch
from src.config import Config
from src.utils import AudioUtil

def check_dataset_structure():
    print("--- 1. Cek Struktur Folder Dataset ---")
    root_dir = "data/labeled"
    labels = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']
    
    total_files = 0
    if not os.path.exists(root_dir):
        print(f"Error: Folder {root_dir} tidak ditemukan!")
        return

    for label in labels:
        path = os.path.join(root_dir, label)
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.wav"))
            count = len(files)
            total_files += count
            print(f"{label}: {count} file audio")
        else:
            print(f"Warning: Folder {label} belum dibuat.")
    
    print(f"Total File Audio: {total_files}")
    if total_files == 0:
        print("Peringatan: Tidak ada data untuk dilatih!")
    print("-" * 30)
    return total_files > 0

def check_preprocessing():
    print("\n--- 2. Cek Preprocessing (Audio -> Spectrogram) ---")
    sample_files = glob.glob("data/labeled/**/*.wav", recursive=True)
    
    if not sample_files:
        print("Tidak ada file wav untuk dites preprocessing.")
        return

    test_file = sample_files[0]
    print(f"Menguji file: {os.path.basename(test_file)}")
    
    try:
        tensor = AudioUtil.preprocess(test_file)
        
        if tensor is not None:
            print(f"Berhasil load & convert!")
            print(f"   Shape Tensor: {tensor.shape}") 
            print("   (Harus [1, N_MELS, Time]) -> Contoh: [1, 128, 130]")
            
                if tensor.shape[0] == 1 and tensor.shape[1] == Config.N_MELS:
                print("   Status: Dimensi Valid untuk ResNet")
            else:
                print("   Status: Dimensi Aneh (Cek config.py)")
        else:
            print("Gagal preprocess (Return None)")
            
    except Exception as e:
        print(f"Error saat preprocessing: {e}")

if __name__ == "__main__":
    if check_dataset_structure():
        check_preprocessing()