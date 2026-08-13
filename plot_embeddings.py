import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.config import Config
from src.model import ContrastiveModel
from src.dataset import FineTuneDataset

DEVICE = Config.DEVICE
MODEL_DIR = Config.MODEL_DIR
FIGURE_DIR = Config.FIGURE_DIR
LABELS = ["MUMTAZ", "JAYYID_JIDDAN", "JAYYID", "MAQBUL", "RASIB"]

print(f"Memulai visualisasi Embedding Space Tahsin di {DEVICE}...")

PRECOMPUTED_LABELED = "/kaggle/working/precomputed_labeled"
if not os.path.exists(PRECOMPUTED_LABELED):
    print(f"Error: Folder {PRECOMPUTED_LABELED} tidak ditemukan.")
    exit()

dataset = FineTuneDataset(PRECOMPUTED_LABELED)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

seed = Config.DEFAULT_SEED
model_path = os.path.join(MODEL_DIR, f"classifier_seed_{seed}.pth")

model = ContrastiveModel(num_classes=len(LABELS), mode='finetune').to(DEVICE)

if os.path.exists(model_path):
    print(f"Load model terbaik dari: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print(f"Model classifier_seed_{seed}.pth belum ada! Pastikan sudah run train.py sampai selesai.")
    exit()

model.eval()

all_embeddings = []
all_labels = []

print("Mengekstrak embeddings dari ResNet18...")
with torch.no_grad():
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        
        features = model.backbone(inputs)
        features = torch.flatten(features, 1)
        
        all_embeddings.append(features.cpu())
        all_labels.extend(targets.numpy())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

print("Menjalankan reduksi dimensi t-SNE (tunggu bentar ya)...")
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'label': [LABELS[i] for i in all_labels]
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set2', s=60, alpha=0.8)
plt.title('t-SNE Projection of Audio Embeddings (Tahsin Quality)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

save_path = os.path.join(FIGURE_DIR, "tsne_embeddings.png")
plt.savefig(save_path)
print(f"Plot embedding berhasil disimpan di {save_path}")
