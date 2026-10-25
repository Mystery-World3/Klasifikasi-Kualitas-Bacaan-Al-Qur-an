import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

sns.set_theme(style="whitegrid")

HISTORY_STAGE1 = "history/history_stage1.csv"
HISTORY_STAGE2 = "history/history_stage2.csv"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURE_DIR, "ssl"), exist_ok=True)

def plot_combined_metrics(df, seed, title_prefix, save_path):
    seed_data = df[df['seed'] == seed] if 'seed' in df.columns else df
    epochs = seed_data['epoch']
    
    plt.figure(figsize=(14, 6))

    # 1. Subplot Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in df.columns:
        plt.plot(epochs, seed_data['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2.5, marker='o')
    if 'val_loss' in df.columns:
        plt.plot(epochs, seed_data['val_loss'], label='Validation Loss', color='#c0392b', linewidth=2.5, marker='s', linestyle='--')
    
    plt.title(f'{title_prefix} - Loss (Seed {seed})', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)

    # 2. Subplot Accuracy
    plt.subplot(1, 2, 2)
    if 'train_acc' in df.columns:
        plt.plot(epochs, seed_data['train_acc'], label='Train Accuracy', color='#3498db', linewidth=2.5, marker='o')
    if 'val_acc' in df.columns:
        plt.plot(epochs, seed_data['val_acc'], label='Validation Accuracy', color='#2980b9', linewidth=2.5, marker='s', linestyle='--')
    
    plt.title(f'{title_prefix} - Accuracy (Seed {seed})', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_history():
    
    # 1. PLOT STAGE 1 (PRE-TRAINING)
    if os.path.exists(HISTORY_STAGE1):
        print("Memproses visualisasi Stage 1 (Pre-Training)...")
        df1 = pd.read_csv(HISTORY_STAGE1)
        seeds1 = df1['seed'].unique()
        
        for seed in seeds1:
            seed_data = df1[df1['seed'] == seed]
            plt.figure(figsize=(8, 6))
            plt.plot(seed_data['epoch'], seed_data['train_loss'], label='Contrastive Loss', color='#8e44ad', linewidth=2.5, marker='o')
            plt.title(f'Pre-Training Loss - Stage 1 (Seed {seed})', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('NT-Xent Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_DIR, f'loss_stage1_seed_{seed}.png'), dpi=300)
            plt.close()
    else:
        print(f"Info: {HISTORY_STAGE1} tidak ditemukan.")

    # 2. PLOT STAGE 2 (FINE-TUNING SSCL)
    if os.path.exists(HISTORY_STAGE2):
        print("Memproses visualisasi Stage 2 (Fine-Tuning SSCL)...")
        df2 = pd.read_csv(HISTORY_STAGE2)
        seeds2 = df2['seed'].unique()
        for seed in seeds2:
            save_path = os.path.join(FIGURE_DIR, f'combined_stage2_seed_{seed}.png')
            plot_combined_metrics(df2, seed, "Fine-Tuning SSCL", save_path)
    else:
        print(f"Info: {HISTORY_STAGE2} tidak ditemukan.")

    # 3. PLOT SSL BASELINE
    ssl_files = glob.glob('history/ssl/ssl_history_seed_*.csv')
    if ssl_files:
        print("Memproses visualisasi SSL Baseline...")
        for file in ssl_files:
            seed_num = int(file.split('seed_')[-1].replace('.csv', ''))
            df_ssl = pd.read_csv(file)
            save_path = os.path.join(FIGURE_DIR, "ssl", f'combined_ssl_seed_{seed_num}.png')
            
            # Pisahkan phase initial dan ssl jika ada
            if 'phase' in df_ssl.columns:
                df_initial = df_ssl[df_ssl['phase'] == 'initial']
                df_pseudo = df_ssl[df_ssl['phase'] == 'ssl']
                
                if not df_initial.empty:
                    plot_combined_metrics(df_initial, seed_num, "SSL Initial Training", os.path.join(FIGURE_DIR, "ssl", f'initial_ssl_seed_{seed_num}.png'))
                if not df_pseudo.empty:
                    plot_combined_metrics(df_pseudo, seed_num, "SSL Pseudo-Label Training", os.path.join(FIGURE_DIR, "ssl", f'pseudo_ssl_seed_{seed_num}.png'))
            else:
                plot_combined_metrics(df_ssl, seed_num, "SSL Baseline", save_path)
    else:
        print("Info: Log SSL Baseline tidak ditemukan.")
        
    print("-" * 50)
    print(f"Selesai! Semua grafik gabungan SSCL dan SSL berhasil disimpan di folder '{FIGURE_DIR}'")

if __name__ == "__main__":
    plot_training_history()