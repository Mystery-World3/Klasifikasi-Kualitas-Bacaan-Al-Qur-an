# train.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.utils import AudioUtil
from src.model import ClassifierModel

# ==========================================
# 1. KONFIGURASI TRAINING
# ==========================================
DATASET_PATH = "data/labeled"                        
MODEL_SAVE_PATH = "models/final_model_skripsi.pth"
LABELS = ['Mumtaz', 'Jayyid Jiddan', 'Jayyid', 'Maqbul', 'Rosib']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

# ==========================================
# 2. CUSTOM DATASET LOADER
# ==========================================
class QuranDataset(Dataset):
    def __init__(self, root_dir):
        self.file_list = []
        self.labels = []
        
        # Scan semua folder sub-direktori
        print(f"🔄 Scanning dataset di: {root_dir}")
        for label_name in LABELS:
            folder_path = os.path.join(root_dir, label_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ Peringatan: Folder '{label_name}' tidak ditemukan!")
                continue
                
            files = glob.glob(os.path.join(folder_path, "*.wav"))
            for f in files:
                self.file_list.append(f)
                self.labels.append(LABEL_MAP[label_name])
                
        print(f"✅ Total data ditemukan: {len(self.file_list)} file.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Preprocessing (Audio -> Spectrogram Tensor)
        # Menggunakan utils.py yang baru
        spec_tensor = AudioUtil.preprocess(audio_path)
        
        # Handling jika audio rusak/gagal load
        if spec_tensor is None:
            # Return tensor kosong (akan di-skip di training loop idealnya, 
            # tapi untuk simpel kita return random zeros)
            spec_tensor = torch.zeros((1, Config.N_MELS, int(Config.N_SAMPLES/Config.HOP_LENGTH)+1))
            
        return spec_tensor, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    # Cek Device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Memulai Training menggunakan: {device}")
    
    # Load Dataset
    dataset = QuranDataset(DATASET_PATH)
    if len(dataset) == 0:
        print("❌ Error: Tidak ada data file .wav ditemukan. Cek folder dataset!")
        return

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Inisialisasi Model ResNet18
    model = ClassifierModel(num_classes=len(LABELS))
    model.to(device)
    model.train() # Mode training
    
    # Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Loop Epoch
    for epoch in range(Config.EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero Gradients
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()
            
            # Statistik Akurasi
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Print progress setiap 5 batch
            if (i+1) % 5 == 0:
                print(f"   Batch {i+1}: Loss {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(f"   👉 Selesai Epoch {epoch+1} | Loss Rata-rata: {running_loss/len(dataloader):.4f} | Akurasi: {epoch_acc:.2f}%")

    # ==========================================
    # 4. SIMPAN MODEL
    # ==========================================
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training Selesai! Model disimpan di: {MODEL_SAVE_PATH}")
    print("Sekarang Anda bisa menjalankan 'app_demo.py' atau 'predict.py'.")

if __name__ == "__main__":
    train()