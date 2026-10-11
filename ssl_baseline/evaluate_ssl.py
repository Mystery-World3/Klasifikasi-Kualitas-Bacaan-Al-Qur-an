

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time

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
    classification_report,
    ConfusionMatrixDisplay,
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

LABEL_MAP = {name: idx for idx, name in enumerate(LABELS)}

MODEL_DIR = os.path.join(Config.MODEL_DIR, "ssl")
RESULT_DIR = os.path.join(Config.RESULT_DIR, "ssl")
FIGURE_DIR = os.path.join(Config.FIGURE_DIR, "ssl")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def get_label_from_path(path):
    filename = os.path.basename(path)

    for class_name, label in LABEL_MAP.items():
        if filename.startswith(class_name):
            return label

    raise ValueError(f"Label tidak dikenali: {filename}")


def load_feature(path):
    # Format loading harus sama dengan training.
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


def create_model(device):
    model = ContrastiveModel(
        num_classes=len(LABELS),
        mode="finetune",
        pretrained=True,
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    inference_seconds = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Sinkronisasi diperlukan agar timing GPU lebih akurat.
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_seconds += time.perf_counter() - start

        pred = output.argmax(dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        total_samples += y.size(0)

    latency_ms = (
        inference_seconds / total_samples * 1000
        if total_samples else float("nan")
    )

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        latency_ms,
    )


def main():
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    all_results = []

    for seed in Config.SEEDS:
        print(f"\n{'=' * 60}")
        print(f"TEST EVALUATION | SEED {seed}")
        print(f"{'=' * 60}")

        set_seed(seed)

        split_path = os.path.join(
            Config.SPLIT_DIR,
            f"seed_{seed}.json",
        )

        if not os.path.isfile(split_path):
            print(f"Split tidak ditemukan: {split_path}")
            continue

        model_path = os.path.join(
            MODEL_DIR,
            f"ssl_classifier_seed_{seed}.pth",
        )

        if not os.path.isfile(model_path):
            print(f"Checkpoint tidak ditemukan: {model_path}")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

        test_files = split["test"]

        if not test_files:
            print("Test set kosong; seed dilewati.")
            continue

        dataset = TestDataset(test_files)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        model = create_model(device)

        checkpoint = torch.load(
            model_path,
            map_location=device,
        )
        model.load_state_dict(checkpoint)

        y_true, y_pred, latency_ms = evaluate(
            model, loader, device
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(
            y_true, y_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
        recall_macro = recall_score(
            y_true, y_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
        f1_macro = f1_score(
            y_true, y_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
        precision_weighted = precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0,
        )
        recall_weighted = recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0,
        )

        print(f"Jumlah test       : {len(dataset)}")
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Precision Macro   : {precision_macro:.4f}")
        print(f"Recall Macro      : {recall_macro:.4f}")
        print(f"Macro F1           : {f1_macro:.4f}")
        print(f"Precision Weighted: {precision_weighted:.4f}")
        print(f"Recall Weighted   : {recall_weighted:.4f}")
        print(f"Weighted F1       : {f1_weighted:.4f}")
        print(f"Latency (ms/audio): {latency_ms:.3f}")

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,
            digits=4,
        )

        print("\nClassification Report")
        print(report)

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

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        ).plot(ax=ax, values_format="d", xticks_rotation=45)

        ax.set_title(f"SSL Test Confusion Matrix - Seed {seed}")
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

        all_results.append({
            "seed": seed,
            "n_test": len(dataset),
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "latency_ms_per_audio": latency_ms,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_results:
        print("Tidak ada hasil evaluasi. Periksa split dan checkpoint.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(RESULT_DIR, "ssl_test_results_all_seeds.csv"),
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "latency_ms_per_audio",
    ]

    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [results_df[col].mean() for col in metric_columns],
        "std": [
            results_df[col].std(ddof=1)
            if len(results_df) > 1 else 0.0
            for col in metric_columns
        ],
    })

    summary.to_csv(
        os.path.join(RESULT_DIR, "ssl_test_summary_mean_std.csv"),
        index=False,
    )

    print(f"\n{'=' * 60}")
    print("SSL TEST RESULTS — ALL SEEDS")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))

    print("\nMean ± Std")
    for _, row in summary.iterrows():
        print(
            f"{row['metric']:24s}: "
            f"{row['mean']:.4f} ± {row['std']:.4f}"
        )

    print(f"\nHasil tersimpan di: {RESULT_DIR}")
    print(f"Confusion matrix di: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
