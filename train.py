# train.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config import set_seed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from src.config import Config
from src.utils import AudioUtil
from src.model import ContrastiveModel

# Konfigurasi
MODEL_DIR = Config.MODEL_DIR
RESULT_DIR = Config.RESULT_DIR
HISTORY_DIR = Config.HISTORY_DIR
FIGURE_DIR = Config.FIGURE_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

class LabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        print(f"Scanning folder labeled di: {root_dir}")
        for label in LABELS:
            path = os.path.join(root_dir, label)
            files = glob.glob(os.path.join(path, "*.wav"))
            for f in files:
                self.data.append((f, LABEL_MAP[label]))
        print(f"Stage 2: Ditemukan {len(self.data)} data labeled.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        tens = AudioUtil.preprocess(path, add_noise=False) # Tidak perlu noise saat fine-tune
        if tens is None: tens = torch.zeros(1, Config.N_MELS, 130)
        return tens, torch.tensor(label, dtype=torch.long)

def train_one_seed(seed):
    device = Config.DEVICE
    print(f"Memulai Stage 2: Fine-Tuning di {device}")
    
    set_seed(seed)

    print("=" * 60)
    print(f"Fine-Tuning Seed : {seed}")
    print("=" * 60)
    
    # 1. Setup Dataset
    dataset = LabeledDataset(Config.LABELED_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"Train : {len(train_dataset)}")
    print(f"Validation : {len(val_dataset)}")

    # 2. Init Model (Mode Finetune)
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune').to(device)
    
    # 3. Load Bobot dari Stage 1 (Transfer Learning)
    PRETRAINED_PATH = os.path.join(
        MODEL_DIR,
        f"encoder_seed_{seed}.pth"
    )

    if os.path.exists(PRETRAINED_PATH):
        print(
            f"Memuat {PRETRAINED_PATH}"
        )

        state_dict = torch.load(
            PRETRAINED_PATH,
            map_location=device
        )

        model.load_state_dict(
            state_dict,
            strict=False
        )

    else:
        print("Encoder tidak ditemukan")
        return

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE
    )
    model.train()
    best_acc = 0
    patience = 0
    history = []
    all_true = []
    all_pred = []
    for epoch in range(Config.EPOCHS_FINETUNE):
        correct = 0
        total = 0
        total_loss = 0
        
        for inputs, targets in tqdm(
            train_loader,
            desc=f"Seed {seed} | Epoch {epoch+1}"
        ):
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
        val_correct = 0
        val_total = 0
        val_loss = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (pred == targets).sum().item()
                all_true.extend(targets.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())

        val_acc = 100 * val_correct / val_total

        precision = precision_score(
            all_true,
            all_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            all_true,
            all_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            all_true,
            all_pred,
            average="weighted",
            zero_division=0
        )

        model.train()
    
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_acc": acc,
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS_FINETUNE}"
            f" | Train Loss : {total_loss/len(train_loader):.4f}"
            f" | Val Loss : {val_loss/len(val_loader):.4f}"
            f" | Train Acc : {acc:.2f}%"
            f" | Val Acc : {val_acc:.2f}%"
            f" | Precision : {precision:.4f}"
            f" | Recall : {recall:.4f}"
            f" | F1 : {f1:.4f}"
        )
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            
            best_acc = val_acc
            
            patience = 0
            
            MODEL_PATH = os.path.join(
                MODEL_DIR,
                f"classifier_seed_{seed}.pth"
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )
            
            print(f"Best Model disimpan : {MODEL_PATH}")

        else:
            patience += 1

        # EARLY STOPPING
        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early Stopping")
            break

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            HISTORY_DIR,
            f"history_seed_{seed}.csv"
        ),
        index=False
    )

    result_df = pd.DataFrame({
        "seed": [seed],
        "best_accuracy": [best_acc],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1]
    })

    result_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"result_seed_{seed}.csv"
        ),
        index=False
    )

    plt.figure(figsize=(8,5))
    plt.plot(
        history_df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history_df["val_loss"],
        label="Validation Loss"
    )
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve Seed {seed}")
    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            f"loss_seed_{seed}.png"
        )
    )

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(
        history_df["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        history_df["val_acc"],
        label="Validation Accuracy"
    )

    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            f"accuracy_seed_{seed}.png"
        )
    )

    plt.close()

if __name__ == "__main__":

    for seed in Config.SEEDS:

        train_one_seed(seed)