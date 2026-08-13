import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from src.config import Config, set_seed
from src.model import ContrastiveModel
from src.dataset import FineTuneDataset

MODEL_DIR = Config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("history", exist_ok=True)
LABELS = ["MUMTAZ", "JAYYID_JIDDAN", "JAYYID", "MAQBUL", "RASIB"]

def train_one_seed(seed):
    device = Config.DEVICE
    target_model_path = os.path.join(Config.MODEL_DIR, f"classifier_seed_{seed}.pth")
    
    if os.path.exists(target_model_path):
        print("="*60)
        print(f"SKIPPING SEED {seed} | Fine-tuned model sudah ada di {target_model_path}")
        print("="*60)
        return None

    print(f"Memulai Stage 2: Fine-Tuning di {device}")
    set_seed(seed)
    
    PRECOMPUTED_LABELED = "/kaggle/working/precomputed_labeled"
    if not os.path.exists(PRECOMPUTED_LABELED):
        print("Error: Folder precomputed tidak ditemukan.")
        return None
        
    dataset = FineTuneDataset(PRECOMPUTED_LABELED)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune').to(device)
    pretrained_path = os.path.join(MODEL_DIR, f"encoder_seed_{seed}.pth")

    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    else:
        print("Encoder tidak ditemukan")
        return None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE)
    
    best_acc = 0
    patience = 0
    seed_history = []
    
    for epoch in range(Config.EPOCHS_FINETUNE):
        model.train()
        correct, total, total_loss = 0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"Seed {seed} | Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            
        acc = 100 * correct / total
        
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, pred = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (pred == targets).sum().item()
                
        val_acc = 100 * val_correct / val_total
        epoch_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        seed_history.append({
            'seed': seed,
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_acc': val_acc
        })
        
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), target_model_path)
        else:
            patience += 1

        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early Stopping")
            break
            
    return seed_history

if __name__ == "__main__":
    all_history = []
    for seed in Config.SEEDS:
        history = train_one_seed(seed)
        if history:
            all_history.extend(history)
            
    if all_history:
        csv_path = "history/history_stage2.csv"
        df = pd.DataFrame(all_history)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Log Fine-Tuning Stage 2 berhasil disimpan ke {csv_path}")