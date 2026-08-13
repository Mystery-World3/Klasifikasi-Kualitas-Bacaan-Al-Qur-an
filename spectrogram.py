import os
import random
import glob
import torch
import matplotlib.pyplot as plt

PRECOMPUTED_DIR = "/kaggle/working/precomputed_labeled"

if not os.path.exists(PRECOMPUTED_DIR):
    print("Folder precomputed tidak ditemukan. Run prepare_dataset.py dulu bro.")
    exit()

pt_files = glob.glob(os.path.join(PRECOMPUTED_DIR, "*.pt"))
if len(pt_files) == 0:
    print("Tidak ada file .pt yang ditemukan.")
    exit()

sample_file = random.choice(pt_files)
tensor = torch.load(sample_file)

plt.figure(figsize=(8, 6))
plt.imshow(tensor.squeeze(0).numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.title(f"Spectrogram (64x64): {os.path.basename(sample_file)}")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
save_path = os.path.join("figures", "sample_spectrogram.png")
plt.savefig(save_path)
print(f"Spectrogram berhasil di-generate dan disimpan di {save_path}")