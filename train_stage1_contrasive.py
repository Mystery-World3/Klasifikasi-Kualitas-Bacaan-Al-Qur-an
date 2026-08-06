# train_stage1_contrastive.py
import os
import glob
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.config import set_seed
from src.utils import AudioUtil
from src.model import ContrastiveModel
from src.loss import NTXentLoss

SAVE_FOLDER = Config.MODEL_DIR

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir):
        # Ambil semua file .wav di folder unlabeled
        self.files = glob.glob(os.path.join(root_dir, "*.wav"))
        print(f"Stage 1: Ditemukan {len(self.files)} data unlabeled.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # View 1: Audio Asli
        xi = AudioUtil.preprocess(file_path, add_noise=False)
        # View 2: Audio + Sedikit Noise (Supaya model belajar bedanya)
        xj = AudioUtil.preprocess(file_path, add_noise=True)

        if xi is None or xj is None:
            dummy = torch.zeros(1, Config.N_MELS, 130)
            return dummy, dummy
            
        return xi, xj

def train_one_seed(seed):
    device = Config.DEVICE
    print(f"Memulai Pre-training (Semi-Supervised) di {device}")
    set_seed(seed)
    print("="*60)
    print(f"Random Seed : {seed}")
    print("="*60)
    if not os.path.exists(Config.UNLABELED_DIR):
        print(f"Error: Folder {Config.UNLABELED_DIR} tidak ditemukan!")
        return

    dataset = UnlabeledDataset(Config.UNLABELED_DIR)
    if len(dataset) < 2:
        print("Error: Butuh minimal 2 file audio di folder unlabeled!")
        return

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Init Model mode 'pretrain'
    model = ContrastiveModel(mode='pretrain').to(device)
    loss_fn = NTXentLoss(Config.BATCH_SIZE, device=device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE
    )
    best_loss = float("inf")
    patience = 0
    history = []
    model.train()
    for epoch in range(
        Config.EPOCHS_PRETRAIN
    ):
        total_loss = 0
        
        for xi, xj in tqdm(
            loader,
            desc=f"Seed {seed} | Epoch {epoch+1}"
        ):            
            
            xi = xi.to(device)
            xj = xj.to(device)

            optimizer.zero_grad()
            zi = model(xi)
            zj = model(xj)
            
            loss = loss_fn(zi, zj)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS_PRETRAIN}] | Contrastive Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        
            torch.save(
                model.state_dict(),
                os.path.join(
                    SAVE_FOLDER,
                    f"encoder_seed_{seed}.pth"
                )
            )
            print(
                f"Best Encoder disimpan : encoder_seed_{seed}.pth"
            )
        else:
            patience += 1
        
        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping...")
            break

    history_df = pd.DataFrame({
        "loss": history
        })
    
    history_df.to_csv(
        os.path.join(
        Config.HISTORY_DIR,
        f"contrastive_seed_{seed}.csv"
        ),
        index=False
        )

    plt.figure(figsize=(8,5))
    plt.plot(history_df["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Contrastive Loss Seed {seed}"
    )
    plt.grid()
    plt.savefig(
        os.path.join(
            Config.FIGURE_DIR,
            f"loss_seed_{seed}.png"
        )
    )

    plt.close()    

if __name__ == "__main__":
    for seed in Config.SEEDS:
        train_one_seed(seed)