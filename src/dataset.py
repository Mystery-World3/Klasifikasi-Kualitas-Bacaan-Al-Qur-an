import os
import glob
import json
import torch
from torch.utils.data import Dataset

class ContrastivePretrainDataset(Dataset):
    def __init__(self, precomputed_data_path):
        self.files = glob.glob(os.path.join(precomputed_data_path, "*.pt"))
        print(f"Jumlah data unlabeled (Pre-computed) : {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def apply_tensor_augment(self, tensor, add_noise=False):
        aug_tensor = tensor.clone()
        shift_amount = torch.randint(-15, 15, (1,)).item()
        aug_tensor = torch.roll(aug_tensor, shifts=shift_amount, dims=2) 
        if add_noise:
            noise = torch.randn_like(aug_tensor) * 0.05
            aug_tensor = aug_tensor + noise
        return aug_tensor

    def __getitem__(self, idx):
        pt_path = self.files[idx]
        try:
            tensor = torch.load(pt_path) 
        except:
            tensor = torch.zeros(1, 64, 64)
            
        view1 = self.apply_tensor_augment(tensor, add_noise=False)
        view2 = self.apply_tensor_augment(tensor, add_noise=True)
        
        return view1, view2


class FineTuneDataset(Dataset):
    def __init__(self, precomputed_data_path):
        self.files = []
        self.labels = []
        self.class_map = {
            "MUMTAZ": 0, "JAYYID_JIDDAN": 1, "JAYYID": 2, "MAQBUL": 3, "RASIB": 4
        }
        pt_files = glob.glob(os.path.join(precomputed_data_path, "*.pt"))
        
        for pt_path in pt_files:
            filename = os.path.basename(pt_path)
            for class_name, label in self.class_map.items():
                if filename.startswith(class_name):
                    self.files.append(pt_path)
                    self.labels.append(label)
                    break
        print(f"Jumlah data labeled (Pre-computed) : {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_path = self.files[idx]
        label = self.labels[idx]
        try:
            spec = torch.load(pt_path)
        except:
            spec = torch.zeros(1, 64, 64)
            
        return spec, torch.tensor(label, dtype=torch.long)

class SplitFineTuneDataset(Dataset):

    def __init__(self, split_json, mode="train"):

        with open(split_json, "r") as f:
            split = json.load(f)

        self.files = split[mode]

        self.label_map = {
            "MUMTAZ": 0,
            "JAYYID_JIDDAN": 1,
            "JAYYID": 2,
            "MAQBUL": 3,
            "RASIB": 4
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        feature = torch.load(path)

        if isinstance(feature, dict):

            if "feature" in feature:
                x = feature["feature"]

            elif "spectrogram" in feature:
                x = feature["spectrogram"]

            else:
                x = list(feature.values())[0]

        else:
            x = feature

        filename = os.path.basename(path)

        label_name = None

        for cls in sorted(self.label_map.keys(), key=len, reverse=True):
            if filename.startswith(cls):
                label_name = cls
                break

        if label_name is None:
            raise ValueError(f"Label tidak dikenali: {filename}")

        y = self.label_map[label_name]

        return x.float(), torch.tensor(y, dtype=torch.long)