import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.config import Config
from src.model import ContrastiveModel
from src.utils import AudioUtil
from train import LabeledDataset, LABELS  # Import dataset dari script train

# SETUP
MODEL_PATH = "models/final_model_skripsi.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    print("📊 Memulai Evaluasi Model...")
    
    # 1. Load Model
    model = ContrastiveModel(num_classes=len(LABELS), mode='finetune')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. Load Data
    dataset = LabeledDataset(Config.LABELED_DIR)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    y_true = []
    y_pred = []

    # 3. Prediksi Massal
    print("🔄 Sedang memproses seluruh data...")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Prediksi AI')
    plt.ylabel('Label Sebenarnya (Dosen)')
    plt.title('Confusion Matrix - Tahsin AI')
    
    # Simpan Gambar untuk Skripsi
    plt.savefig('gambar_confusion_matrix.png')
    print("✅ Gambar 'gambar_confusion_matrix.png' berhasil disimpan!")
    
    # 5. Print Laporan Statistik
    print("\n" + "="*50)
    print("HASIL EVALUASI SKRIPSI (Copy ke Bab 4)")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=LABELS))

if __name__ == "__main__":
    evaluate()