# src/dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
from src.utils import AudioUtil
from src.config import Config

class ContrastivePretrainDataset(Dataset):
    """
    Dataset untuk Fase Pre-training (Unlabeled).
    Output: (View1, View2) -> Dua versi augmentasi dari audio yang sama.
    """
    def __init__(self, data_path):
        # Mencari semua file .wav di folder
        self.files = glob.glob(os.path.join(data_path, "*.wav"))
        if len(self.files) == 0:
            print(f"Peringatan: Tidak ada file .wav ditemukan di {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        
        # 1. Load Audio
        sig = AudioUtil.open_audio(audio_file, Config.SAMPLE_RATE)
        sig = AudioUtil.rechannel(sig)
        sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
        
        # 2. Buat View 1 (Augmentasi Pertama)
        # Misal: Geser waktu
        aug1 = AudioUtil.time_shift(sig)
        aug1 = torch.tensor(aug1, dtype=torch.float32)

        # 3. Buat View 2 (Augmentasi Kedua)
        # Misal: Tambah noise + Geser waktu
        aug2 = AudioUtil.add_noise(sig)
        aug2 = AudioUtil.time_shift(aug2) # Augmentasi bertumpuk
        aug2 = torch.tensor(aug2, dtype=torch.float32)

        return aug1, aug2

class FineTuneDataset(Dataset):
    """
    Dataset untuk Fase Klasifikasi (Labeled).
    Output: (Audio, Label)
    """
    def __init__(self, data_path):
        self.files = []
        self.labels = []
        # Mapping label folder ke angka
        self.class_map = {
            'mumtaz': 0, 'jayyid_jiddan': 1, 'jayyid': 2, 'maqbul': 3, 'rosib': 4
        }
        
        # Loop setiap folder kelas
        for class_name, label_idx in self.class_map.items():
            folder_path = os.path.join(data_path, class_name)
            wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
            self.files.extend(wav_files)
            self.labels.extend([label_idx] * len(wav_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        label = self.labels[idx]
        
        # Load Audio (Tanpa Augmentasi ekstrim saat testing, cukup standardisasi)
        sig = AudioUtil.open_audio(audio_file, Config.SAMPLE_RATE)
        sig = AudioUtil.rechannel(sig)
        sig = AudioUtil.pad_trunc(sig, Config.N_SAMPLES)
        
        return torch.tensor(sig, dtype=torch.float32), torch.tensor(label, dtype=torch.long)