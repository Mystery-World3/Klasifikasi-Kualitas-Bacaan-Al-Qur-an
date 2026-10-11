
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.config import Config, set_seed
from src.model import ContrastiveModel




LABELS = [
    "MUMTAZ",
    "JAYYID_JIDDAN",
    "JAYYID",
    "MAQBUL",
    "RASIB",
]

LABEL_MAP = {
    label: idx for idx, label in enumerate(LABELS)
}

MODEL_DIR = os.path.join(Config.MODEL_DIR, "ssl")
RESULT_DIR = os.path.join(Config.RESULT_DIR, "ssl")
FIGURE_DIR = os.path.join(Config.FIGURE_DIR, "ssl")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = torch.device(Config.DEVICE)




def get_label_from_path(path):
    filename = os.path.basename(path)

    for class_name, label in LABEL_MAP.items():
        if filename.startswith(class_name):
            return label

    raise ValueError(
        f"Label tidak dikenali dari nama file: {filename}"
    )


def load_feature(path):
    feature = torch.load(path, map_location="cpu")

    if isinstance(feature, dict):
        if "feature" in feature:
            feature = feature["feature"]
        elif "spectrogram" in feature:
            feature = feature["spectrogram"]
        else:
            raise ValueError(
                f"Key fitur tidak dikenali pada: {path}"
            )

    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"Fitur bukan Tensor: {path}")

    return feature.float()


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.samples = [
            (path, get_label_from_path(path))
            for path in test_files
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feature = load_feature(path)

        return feature, label



@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)

        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)

        y_true.extend(targets.numpy())
        y_pred.extend(predictions.cpu().numpy())

    return np.asarray(y_true), np.asarray(y_pred)


def evaluate():
    print("Memulai Evaluasi Model SSL...")
    print(f"Device: {DEVICE}")

    all_result = []

    for seed in Config.SEEDS:
        print("=" * 60)
        print(f"Evaluasi Seed {seed}")
        print("=" * 60)

        set_seed(seed)

        
        model_path = os.path.join(
            MODEL_DIR,
            f"ssl_classifier_seed_{seed}.pth",
        )

        if not os.path.exists(model_path):
            print(f"Model {model_path} tidak ditemukan!")
            continue

        
        split_json = os.path.join(
            Config.SPLIT_DIR,
            f"seed_{seed}.json",
        )

        if not os.path.exists(split_json):
            print(f"Split {split_json} tidak ditemukan!")
            continue

        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)

        test_files = split["test"]

        if not test_files:
            print("Data test kosong, seed dilewati.")
            continue

        dataset = TestDataset(test_files)

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )


        model = ContrastiveModel(
            num_classes=len(LABELS),
            mode="finetune",
        )

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()

        print(f"Jumlah data test : {len(dataset)}")
        print("Sedang memproses data TEST...")

        y_true, y_pred = evaluate_model(model, loader)


        acc = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")


        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,
        )

        print("\nClassification Report")
        print(report)

        # Simpan laporan per seed sebagai file tambahan
        report_path = os.path.join(
            RESULT_DIR,
            f"classification_report_seed_{seed}.txt",
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)


        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            xticks_rotation=45,
        )

        ax.set_title(f"Confusion Matrix - Seed {seed}")

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                FIGURE_DIR,
                f"confusion_matrix_seed_{seed}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


        all_result.append({
            "seed": seed,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


    if all_result:
        result_df = pd.DataFrame(
            all_result,
            columns=[
                "seed",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )

        result_path = os.path.join(
            RESULT_DIR,
            "summary_result.csv",
        )

        result_df.to_csv(result_path, index=False)

        print("=" * 60)
        print("HASIL EVALUASI SSL — SEMUA SEED")
        print("=" * 60)
        print(result_df.to_string(index=False))

        print("\nRata-rata hasil evaluasi:")
        print(
            result_df[
                ["accuracy", "precision", "recall", "f1"]
            ].mean().to_string()
        )

        print(f"\nCSV hasil evaluasi : {result_path}")
        print(f"Confusion matrix   : {FIGURE_DIR}")
        print(f"Classification report: {RESULT_DIR}")

    else:
        print(
            "Tidak ada hasil evaluasi. "
            "Periksa split dan checkpoint model SSL."
        )


if __name__ == "__main__":
    evaluate()
