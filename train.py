# train.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.utils import AudioUtil
from src.model import ContrastiveModel

# Konfigurasi
PRETRAINED_PATH = "models/encoder_pretrained.pth"
FINAL_MODEL_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

class LabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        print(f"🔄 Scanning folder labeled di: {root_dir}")
        for label in LABELS:
            path = os.path.join(root_dir, label)
            files = glob.glob(os.path.join(path, "*.wav"))
            for f in files:
                self.data.append((f, LABEL_MAP[label]))
        print(f"📊 Stage 2: Ditemukan {len(self.data)} data labeled.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        tens = AudioUtil.preprocess(path, add_noise=False) # Tidak perlu noise saat fine-tune
        if tens is None: tens = torch.zeros(1, Config.N_MELS, 130)
        return tens, torch.tensor(label, dtype=torch.long)

def train_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Memulai Stage 2: Fine-Tuning di {device}")
    
    # 1. Setup Dataset
    dataset = LabeledDataset(Config.LABELED_DIR)
    if len(dataset) == 0:
        print("❌ Error: Data Labeled kosong. Cek folder data/labeled!")
        return
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Init Model (Mode Finetune)
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune').to(device)
    
    # 3. Load Bobot dari Stage 1 (Transfer Learning)
    if os.path.exists(PRETRAINED_PATH):
        print("📥 Memuat bobot hasil Pre-training Contrastive Learning...")
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        # strict=False karena kita mau load Backbone-nya saja, Classifier head-nya baru
        model.load_state_dict(state_dict, strict=False) 
    else:
        print("⚠️ Warning: File pretrain tidak ditemukan. Training dari nol (Supervised Murni).")

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    model.train()
    for epoch in range(Config.EPOCHS_FINETUNE):
        correct = 0
        total = 0
        total_loss = 0
        
        for inputs, targets in loader:
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
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Akurasi: {acc:.2f}%")
        
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅ Model Final Siap! Disimpan di: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train_finetune()