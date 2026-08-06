# src/dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
from src.utils import AudioUtil
from src.config import Config

class ContrastivePretrainDataset(Dataset):
    """
    Dataset untuk Stage 1 (Semi-Supervised Contrastive Learning)
    Menghasilkan dua augmentasi dari audio yang sama.
    """

    def __init__(self, data_path):

        self.files = glob.glob(os.path.join(data_path, "*.wav"))

        print(f"Jumlah data unlabeled : {len(self.files)}")

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        audio_path = self.files[idx]

        view1 = AudioUtil.preprocess(
            audio_path,
            add_noise=False,
            shift=True
        )

        view2 = AudioUtil.preprocess(
            audio_path,
            add_noise=True,
            shift=True
        )

        if view1 is None or view2 is None:

            dummy = torch.zeros(1, Config.N_MELS, 130)

            return dummy, dummy

        return view1, view2

class FineTuneDataset(Dataset):
    def __init__(self, data_path):

        self.files = []

        self.labels = []

        self.class_map = {

            "mumtaz":0,
            "jayyid_jiddan":1,
            "jayyid":2,
            "maqbul":3,
            "rosib":4

        }

        for class_name, label in self.class_map.items():

            folder = os.path.join(data_path, class_name)

            wav_files = glob.glob(
                os.path.join(folder, "*.wav")
            )

            self.files.extend(wav_files)

            self.labels.extend(
                [label] * len(wav_files)
            )

        print(f"Jumlah data labeled : {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        audio_path = self.files[idx]
        label = self.labels[idx]

        spec = AudioUtil.preprocess(
            audio_path,
            add_noise=False,
            shift=False
        )

        if spec is None:
            spec = torch.zeros(1, Config.N_MELS, 130)
        return spec, torch.tensor(
        label,
        dtype=torch.long
        )