import os
import glob
import torch
from tqdm import tqdm
from src.config import Config
from src.utils import AudioUtil

def precompute_dataset(input_dir, output_dir, add_noise=False):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    print(f"Menemukan {len(wav_files)} file audio. Memulai pre-compute...")
    
    for wav_path in tqdm(wav_files, desc="Pre-computing"):
        relative_path = os.path.relpath(wav_path, input_dir)
        file_name = relative_path.replace(os.sep, "_").replace(".wav", ".pt")
        save_path = os.path.join(output_dir, file_name)
        
        tensor_data = AudioUtil.preprocess(wav_path, add_noise=add_noise, shift=False)
        
        if tensor_data is not None:
            torch.save(tensor_data, save_path)

if __name__ == "__main__":
    print("=== Pre-compute Labeled Data (Stage 2) ===")
    PRECOMPUTED_LABELED = "/kaggle/working/precomputed_labeled"
    precompute_dataset(Config.LABELED_DIR, PRECOMPUTED_LABELED, add_noise=False)
    
    print("\n=== Pre-compute Unlabeled Data (Stage 1) ===")
    PRECOMPUTED_UNLABELED = "/kaggle/working/precomputed_unlabeled"
    precompute_dataset(Config.UNLABELED_DIR, PRECOMPUTED_UNLABELED, add_noise=False)