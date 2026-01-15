# train_stage1_contrastive.py
import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.utils import AudioUtil
from src.model import ContrastiveModel
from src.loss import NTXentLoss

SAVE_PATH = "models/encoder_pretrained.pth"

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir):
        # Ambil semua file .wav di folder unlabeled
        self.files = glob.glob(os.path.join(root_dir, "*.wav"))
        print(f"📊 Stage 1: Ditemukan {len(self.files)} data unlabeled.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # SIMCLR MAGIC: Satu gambar diubah jadi 2 versi (Augmentasi)
        # View 1: Audio Asli
        xi = AudioUtil.preprocess(file_path, add_noise=False)
        # View 2: Audio + Sedikit Noise (Supaya model belajar bedanya)
        xj = AudioUtil.preprocess(file_path, add_noise=True)

        if xi is None or xj is None:
            # Return dummy jika error
            dummy = torch.zeros(1, Config.N_MELS, 130)
            return dummy, dummy
            
        return xi, xj

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Memulai Pre-training (Semi-Supervised) di {device}")
    
    # Cek Folder
    if not os.path.exists(Config.UNLABELED_DIR):
        print(f"❌ Error: Folder {Config.UNLABELED_DIR} tidak ditemukan!")
        return

    dataset = UnlabeledDataset(Config.UNLABELED_DIR)
    if len(dataset) < 2:
        print("❌ Error: Butuh minimal 2 file audio di folder unlabeled!")
        return

    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Init Model mode 'pretrain'
    model = ContrastiveModel(mode='pretrain').to(device)
    loss_fn = NTXentLoss(Config.BATCH_SIZE, device=device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    model.train()
    for epoch in range(Config.EPOCHS_PRETRAIN):
        total_loss = 0
        for xi, xj in loader:
            xi, xj = xi.to(device), xj.to(device)
            
            optimizer.zero_grad()
            zi = model(xi)
            zj = model(xj)
            
            loss = loss_fn(zi, zj)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS_PRETRAIN}] | Contrastive Loss: {avg_loss:.4f}")

    # Simpan Otak Model (Encoder)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Stage 1 Selesai! Model disimpan di {SAVE_PATH}")

if __name__ == "__main__":
    train()