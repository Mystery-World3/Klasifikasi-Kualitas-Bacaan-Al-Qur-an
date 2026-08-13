import os
import glob
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.config import Config, set_seed
from src.model import ContrastiveModel
from src.loss import NTXentLoss
from src.dataset import ContrastivePretrainDataset

SAVE_FOLDER = Config.MODEL_DIR
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs("history", exist_ok=True)

def train_one_seed(seed):
    device = Config.DEVICE
    target_model_path = os.path.join(Config.MODEL_DIR, f"encoder_seed_{seed}.pth")
    
    if os.path.exists(target_model_path):
        print("="*60)
        print(f"SKIPPING SEED {seed} | Pre-trained model sudah ada di {target_model_path}")
        print("="*60)
        return None
        
    print(f"Memulai Pre-training (Semi-Supervised) di {device}")
    set_seed(seed)
    
    PRECOMPUTED_UNLABELED = "/kaggle/working/precomputed_unlabeled"
    if not os.path.exists(PRECOMPUTED_UNLABELED):
        print(f"Error: Folder {PRECOMPUTED_UNLABELED} tidak ditemukan! Run prepare_dataset.py dulu.")
        return None

    dataset = ContrastivePretrainDataset(PRECOMPUTED_UNLABELED)
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    
    model = ContrastiveModel(mode='pretrain').to(device)
    loss_fn = NTXentLoss(Config.BATCH_SIZE, device=device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE)
    
    best_loss = float("inf")
    patience = 0
    seed_history = []
    
    for epoch in range(Config.EPOCHS_PRETRAIN):
        model.train()
        total_loss = 0
        for xi, xj in tqdm(loader, desc=f"Seed {seed} | Epoch {epoch+1}"):
            xi, xj = xi.to(device), xj.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xi), model(xj))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        seed_history.append({
            'seed': seed,
            'epoch': epoch + 1,
            'train_loss': avg_loss
        })
        
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS_PRETRAIN}] | Contrastive Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(SAVE_FOLDER, f"encoder_seed_{seed}.pth"))
        else:
            patience += 1
        
        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping...")
            break
            
    return seed_history

if __name__ == "__main__":
    all_history = []
    for seed in Config.SEEDS:
        history = train_one_seed(seed)
        if history:
            all_history.extend(history)
            
    if all_history:
        csv_path = "history/history_stage1.csv"
        df = pd.DataFrame(all_history)
        # Jika file sudah ada, append tanpa header. Jika belum, buat baru dengan header.
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Log Pre-training Stage 1 berhasil disimpan ke {csv_path}")