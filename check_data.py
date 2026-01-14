# check_data.py
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import ContrastivePretrainDataset, FineTuneDataset

def test_pretrain_loader():
    print("--- Cek Dataset Pre-training (Unlabeled) ---")
    dataset = ContrastivePretrainDataset(Config.UNLABELED_DIR)
    
    if len(dataset) == 0:
        print("Data kosong! Pastikan ada file .wav di folder data/unlabeled")
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Take 1 batch
    view1, view2 = next(iter(loader))
    
    print(f"Jumlah Data: {len(dataset)}")
    print(f"Shape View 1: {view1.shape} (Batch, Channel, Samples)")
    print(f"Shape View 2: {view2.shape}")
    
    # Contrastive Validation: View 1 and View 2 must not have exactly the same content
    is_same = torch.allclose(view1, view2)
    print(f"Apakah View 1 persis sama dengan View 2? {is_same} (Harusnya False)")
    print("Sukses!\n")

def test_finetune_loader():
    print("--- Cek Dataset Fine-tuning (Labeled) ---")
    dataset = FineTuneDataset(Config.LABELED_DIR)
    
    if len(dataset) == 0:
        print("Data kosong! Pastikan struktur folder data/labeled/{kelas}/*.wav benar")
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    audio, label = next(iter(loader))
    
    print(f"Jumlah Data: {len(dataset)}")
    print(f"Shape Audio: {audio.shape}")
    print(f"Contoh Label: {label}")
    print("Sukses!\n")

if __name__ == "__main__":
   
    test_pretrain_loader()
    test_finetune_loader()