import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from torch.utils.data import DataLoader
from src.config import Config
from src.model import ContrastiveModel
from src.utils import AudioUtil
from train import LabeledDataset, LABELS  

# SETUP
MODEL_DIR = Config.MODEL_DIR
RESULT_DIR = Config.RESULT_DIR
FIGURE_DIR = Config.FIGURE_DIR

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    print("Memulai Evaluasi Model...")
    
    all_result = []

    for seed in Config.SEEDS:

        print("=" * 60)
        print(f"Evaluasi Seed {seed}")
        print("=" * 60)

        MODEL_PATH = os.path.join(
            MODEL_DIR,
            f"classifier_seed_{seed}.pth"
        )
        
        if not os.path.exists(MODEL_PATH):
            print(f"Model {MODEL_PATH} tidak ditemukan!")
            continue    
        
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
        print("Sedang memproses seluruh data...")
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
            
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            
        acc = accuracy_score(
                    y_true,
                    y_pred
                )
        
        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
        
        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
        
        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # 4. Confision Matrix

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS
        )

        fig, ax = plt.subplots(figsize=(8,6))

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        plt.title(
            f"Confusion Matrix - Seed {seed}"
        )

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                f"confusion_matrix_seed_{seed}.png"
            )
        )

        plt.close()

        print(f"Confusion Matrix Seed {seed} berhasil disimpan.")
    
        # 5. Print Laporan Statistik
        print("\n" + "="*50)
        print("HASIL EVALUASI SKRIPSI (Copy ke Bab 4)")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=LABELS))

        all_result.append({
            "seed": seed,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    result_df = pd.DataFrame(all_result)
        
    result_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "summary_result.csv"
        ),
        index=False
    )

    print("=" * 60)
    print(result_df)
    print("=" * 60)
    print("Mean Accuracy :", result_df["accuracy"].mean())
    print("Std Accuracy :", result_df["accuracy"].std())
    print("Mean Precision :", result_df["precision"].mean())
    print("Mean Recall :", result_df["recall"].mean())
    print("Mean F1 :", result_df["f1"].mean())

if __name__ == "__main__":
    evaluate()