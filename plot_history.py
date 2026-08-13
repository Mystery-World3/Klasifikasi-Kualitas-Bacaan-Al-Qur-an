import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_theme(style="whitegrid")

HISTORY_STAGE1 = "history/history_stage1.csv"
HISTORY_STAGE2 = "history/history_stage2.csv"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_training_history():
    
    # 1. PLOT STAGE 1 (PRE-TRAINING)
    if os.path.exists(HISTORY_STAGE1):
        print("Memproses visualisasi Stage 1 (Pre-Training)...")
        df1 = pd.read_csv(HISTORY_STAGE1)
        seeds1 = df1['seed'].unique()
        
        for seed in seeds1:
            seed_data = df1[df1['seed'] == seed]
            epochs = seed_data['epoch']
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, seed_data['train_loss'], label='Contrastive Loss', color='#8e44ad', linewidth=2.5, marker='o')
            plt.title(f'Pre-Training Loss per Epoch - Stage 1 (Seed {seed})', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('NT-Xent Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            filename = os.path.join(FIGURE_DIR, f'loss_stage1_seed_{seed}.png')
            plt.savefig(filename, dpi=300)
            plt.close()
    else:
        print(f"Info: {HISTORY_STAGE1} tidak ditemukan, skip plot Stage 1.")

    # 2. PLOT STAGE 2 (FINE-TUNING)
    if os.path.exists(HISTORY_STAGE2):
        print("\nMemproses visualisasi Stage 2 (Fine-Tuning)...")
        df2 = pd.read_csv(HISTORY_STAGE2)
        seeds2 = df2['seed'].unique()
        
        for seed in seeds2:
            seed_data = df2[df2['seed'] == seed]
            epochs = seed_data['epoch']
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, seed_data['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2.5, marker='o')
            plt.title(f'Fine-Tuning Loss per Epoch - Stage 2 (Seed {seed})', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('CrossEntropy Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            loss_filename = os.path.join(FIGURE_DIR, f'loss_stage2_seed_{seed}.png')
            plt.savefig(loss_filename, dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, seed_data['val_acc'], label='Validation Accuracy', color='#2980b9', linewidth=2.5, marker='s')
            plt.title(f'Validation Accuracy per Epoch - Stage 2 (Seed {seed})', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.ylim(0, 100) # Fix skala Y dari 0 sampai 100%
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            acc_filename = os.path.join(FIGURE_DIR, f'accuracy_stage2_seed_{seed}.png')
            plt.savefig(acc_filename, dpi=300)
            plt.close()
    else:
        print(f"Info: {HISTORY_STAGE2} tidak ditemukan, skip plot Stage 2.")
        
    print("-" * 50)
    print(f"Selesai bro! Semua grafik Stage 1 dan Stage 2 berhasil disimpan di folder '{FIGURE_DIR}'")

if __name__ == "__main__":
    plot_training_history()