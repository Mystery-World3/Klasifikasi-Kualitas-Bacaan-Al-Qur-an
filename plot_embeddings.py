import torch
import torch.nn as nn
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model_skripsi.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "labeled")
OUTPUT_IMG = os.path.join(BASE_DIR, "Plot_embeddings.png")

from src.utils import AudioUtil
from src.model import ContrastiveModel

def plot_tsne():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Memulai visualisasi Embedding Space Tahsin di {device}...")

    # 1. Cek Folder Data
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Folder {DATA_DIR} tidak ditemukan.")
        return

    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    if len(CLASSES) == 0:
        print(f"❌ Error: Tidak ada folder kategori di dalam {DATA_DIR}.")
        return
        
    print(f"ℹ️  Kategori yang diproses: {CLASSES}")

    # 2. Load Model Final
    model = ContrastiveModel(num_classes=len(CLASSES), mode='finetune').to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"📥 Memuat model: {os.path.basename(MODEL_PATH)}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(f"❌ Error: File model {MODEL_PATH} belum ada. Latih model dulu.")
        return

    model.classifier_head = nn.Identity()
    print("✅ Layer classifier_head di-reset untuk ekstraksi fitur.")

    # 3. Mengumpulkan Data & Ekstrak Fitur
    embeddings = []
    labels = []

    print(f"📂 Memproses file audio Tahsin...")

    for cat in CLASSES:
        files = glob.glob(os.path.join(DATA_DIR, cat, "*.wav"))
        print(f"⏳ Mengekstrak {len(files)} file dari kategori: {cat}...")
        
        # ✅ BENAR: Harus menjorok ke dalam, biar dieksekusi di SETIAP folder kategori
        with torch.no_grad():
            for f_path in files:
                img_tensor = AudioUtil.preprocess(f_path)
                
                # Jaga-jaga kalau ada file audio yang rusak/corrupt biar gak error
                if img_tensor is None:
                    continue 
                    
                img = img_tensor.unsqueeze(0).to(device)
                fitur = model(img)
                embeddings.append(fitur.cpu().numpy().flatten())
                labels.append(cat)
                
    embeddings = np.array(embeddings)
    
    if embeddings.shape[0] == 0:
        print("❌ Tidak ada data audio (.wav) yang berhasil diekstrak.")
        return

    print(f"\n📊 Total Ekstraksi: {embeddings.shape[0]} data. Dimensi vektor: {embeddings.shape[1]}")

    # 4. Mengitung t-SNE
    print("🎨 Menghitung t-SNE (Reduksi Dimensi ke 2D)...")
    nilai_perplexity = min(15, embeddings.shape[0] - 1) 
    tsne = TSNE(n_components=2, random_state=42, perplexity=nilai_perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 5. Menggambar Plot
    print("✍️ Menggambar Grafik Plot...")
    plt.figure(figsize=(12, 10))
    
    color_mapping = {
        'mumtaz': '#2ecc71',        
        'jayyid_jiddan': '#3498db', 
        'jayyid': '#f1c40f',        
        'maqbul': '#e67e22',        
        'rosib': '#e74c3c'          
    }
    
    palette = [color_mapping.get(c, '#95a5a6') for c in CLASSES]
    
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=labels,
        palette=palette,
        hue_order=CLASSES,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )
    
    plt.title('Visualisasi t-SNE: Embedding Kualitas Bacaan Al-Qur\'an', fontsize=16, pad=20)
    plt.xlabel('t-SNE Dimension 1 (Fitur Laten)', fontsize=12)
    plt.ylabel('t-SNE Dimension 2 (Fitur Laten)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Kategori Tahsin")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 6. Simpan Gambar
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n✅ Selesai! Visualisasi disimpan di: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_tsne()