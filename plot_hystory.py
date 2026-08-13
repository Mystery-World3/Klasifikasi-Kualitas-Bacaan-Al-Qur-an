import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_theme(style="whitegrid")

HISTORY_STAGE2 = "history/history_stage2.csv"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_training_history():
    if not os.path.exists(HISTORY_STAGE2):
        print(f"Error: File {HISTORY_STAGE2} belum ada. Pastikan training Stage 2 (train.py) sudah selesai dijalankan.")
        return

    df = pd.read_csv(HISTORY_STAGE2)
    
    seeds = df['seed'].unique()

    for seed in seeds:
        print(f"Membuat visualisasi untuk Seed {seed}...")
        
        # Filter data hanya untuk seed yang sedang diproses
        seed_data = df[df['seed'] == seed]
        epochs = seed_data['epoch']
        
        # 1. PLOT TRAIN LOSS
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, seed_data['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2.5, marker='o')
        plt.title(f'Training Loss per Epoch - Stage 2 (Seed {seed})', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        loss_filename = os.path.join(FIGURE_DIR, f'loss_seed_{seed}.png')
        plt.savefig(loss_filename, dpi=300)
        plt.close()

        # 2. PLOT VALIDATION ACCURACY
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, seed_data['val_acc'], label='Validation Accuracy', color='#2980b9', linewidth=2.5, marker='s')
        plt.title(f'Validation Accuracy per Epoch - Stage 2 (Seed {seed})', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 100) # Fix skala Y dari 0 sampai 100%
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        acc_filename = os.path.join(FIGURE_DIR, f'accuracy_seed_{seed}.png')
        plt.savefig(acc_filename, dpi=300)
        plt.close()
        
    print("-" * 50)
    print(f"Selesai bro! Semua grafik berhasil disimpan di folder '{FIGURE_DIR}'")

if __name__ == "__main__":
    plot_training_history()