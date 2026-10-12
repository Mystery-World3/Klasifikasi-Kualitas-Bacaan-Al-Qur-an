
import os
import sys
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
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
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}

MODEL_DIR = os.path.join(Config.MODEL_DIR, "ssl")
RESULT_DIR = os.path.join(Config.RESULT_DIR, "ssl")
FIGURE_DIR = os.path.join(Config.FIGURE_DIR, "ssl")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = torch.device(Config.DEVICE)


def get_label_from_path(path):
    filename = os.path.basename(path)

    # Urutkan nama kelas dari yang terpanjang agar
    # JAYYID_JIDDAN tidak terbaca sebagai JAYYID.
    for name in sorted(LABEL_MAP, key=len, reverse=True):
        if filename.startswith(name):
            return LABEL_MAP[name]

    raise ValueError(f"Label tidak dikenali: {filename}")


def load_feature(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "feature" in obj:
            obj = obj["feature"]
        elif "spectrogram" in obj:
            obj = obj["spectrogram"]
        else:
            raise ValueError(
                f"Key fitur tidak ditemukan pada file: {path}"
            )

    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Fitur bukan Tensor: {path}")

    return obj.float()


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.samples = [
            (path, get_label_from_path(path))
            for path in test_files
        ]

        missing = [
            path for path, _ in self.samples
            if not os.path.isfile(path)
        ]
        if missing:
            examples = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} file test tidak ditemukan.\n"
                f"Contoh:\n{examples}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return load_feature(path), label


@torch.inference_mode()
def evaluate_model(model, loader, seed):
    model.eval()
    y_true, y_pred = [], []

    progress = tqdm(
        loader,
        desc=f"Seed {seed} - batch test",
        unit="batch",
        dynamic_ncols=True,
    )

    for inputs, targets in progress:
        inputs = inputs.to(DEVICE, non_blocking=True)

        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)

        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    return np.asarray(y_true), np.asarray(y_pred)


def save_summary(rows):
    # Simpan progres setelah setiap seed, bukan hanya di akhir.
    df = pd.DataFrame(
        rows,
        columns=["seed", "accuracy", "precision", "recall", "f1"],
    )
    output_path = os.path.join(RESULT_DIR, "summary_result.csv")
    df.to_csv(output_path, index=False)
    return output_path


def evaluate(seeds, batch_size, num_workers, skip_done):
    print(f"Device       : {DEVICE}", flush=True)
    print(f"Seeds        : {seeds}", flush=True)
    print(f"Batch size   : {batch_size}", flush=True)
    print(f"Num workers  : {num_workers}", flush=True)

    summary_path = os.path.join(RESULT_DIR, "summary_result.csv")
    rows = []

    # Izinkan melanjutkan seed yang sudah selesai.
    if skip_done and os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        if "seed" in old_df.columns:
            old_df = old_df[old_df["seed"].isin(seeds)]
            rows = old_df.to_dict("records")

    completed = {int(row["seed"]) for row in rows}

    for seed in seeds:
        if seed in completed:
            print(f"Seed {seed} sudah ada di CSV, dilewati.", flush=True)
            continue

        start = time.time()
        print(f"\n{'=' * 55}", flush=True)
        print(f"Memulai evaluasi seed {seed}", flush=True)
        print(f"{'=' * 55}", flush=True)

        set_seed(seed)

        model_path = os.path.join(
            MODEL_DIR, f"ssl_classifier_seed_{seed}.pth"
        )
        split_path = os.path.join(
            Config.SPLIT_DIR, f"seed_{seed}.json"
        )

        if not os.path.isfile(model_path):
            print(f"Checkpoint tidak ditemukan: {model_path}", flush=True)
            continue

        if not os.path.isfile(split_path):
            print(f"File split tidak ditemukan: {split_path}", flush=True)
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

        test_files = split.get("test", [])
        if not test_files:
            print(f"Data test seed {seed} kosong, dilewati.", flush=True)
            continue

        dataset = TestDataset(test_files)
        print(f"Jumlah data test: {len(dataset)}", flush=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(DEVICE.type == "cuda"),
        )

        model = ContrastiveModel(
            num_classes=len(LABELS),
            mode="finetune",
            pretrained=False,
        )

        try:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=True)
        model.to(DEVICE)

        print("Checkpoint dimuat. Evaluasi dimulai...", flush=True)
        y_true, y_pred = evaluate_model(model, loader, seed)

        metrics = {
            "seed": seed,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        print("\nMetrik:", flush=True)
        for key, value in metrics.items():
            if key != "seed":
                print(f"{key:10s}: {value:.4f}", flush=True)

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,
        )

        report_path = os.path.join(
            RESULT_DIR, f"classification_report_seed_{seed}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=LABELS
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(
            ax=ax, cmap="Blues", values_format="d",
            xticks_rotation=45
        )
        ax.set_title(f"Confusion Matrix - Seed {seed}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(FIGURE_DIR, f"confusion_matrix_seed_{seed}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        rows.append(metrics)
        summary_path = save_summary(rows)

        print(f"Seed {seed} selesai dalam {(time.time() - start) / 60:.2f} menit.", flush=True)
        print(f"CSV sementara tersimpan: {summary_path}", flush=True)

        del model, loader, dataset, checkpoint, y_true, y_pred
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        df = pd.DataFrame(rows).sort_values("seed")
        save_summary(df.to_dict("records"))
        print("\nRINGKASAN EVALUASI SSL", flush=True)
        print(df.to_string(index=False), flush=True)
        print("\nRata-rata metrik:", flush=True)
        print(
            df[["accuracy", "precision", "recall", "f1"]]
            .mean()
            .to_string(),
            flush=True,
        )
        print(f"\nHasil CSV: {summary_path}", flush=True)
    else:
        print("Belum ada seed yang berhasil dievaluasi.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=Config.SEEDS,
        help="Contoh: --seeds 42 untuk tes satu seed",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Lewati seed yang sudah tercatat di summary_result.csv",
    )
    args = parser.parse_args()

    evaluate(
        seeds=args.seeds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_done=args.skip_done,
    )
