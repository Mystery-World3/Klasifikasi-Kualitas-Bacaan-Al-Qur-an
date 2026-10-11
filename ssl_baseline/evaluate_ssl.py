import os
import glob

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

from src.config import Config, set_seed
from src.utils import AudioUtil
from src.model import ContrastiveModel


SSL_MODEL_DIR = os.path.join(
    Config.MODEL_DIR,
    "ssl"
)

SSL_RESULT_DIR = os.path.join(
    Config.RESULT_DIR,
    "ssl"
)

SSL_FIGURE_DIR = os.path.join(
    Config.FIGURE_DIR,
    "ssl"
)

os.makedirs(
    SSL_RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    SSL_FIGURE_DIR,
    exist_ok=True
)


LABELS = [
    "Mumtaz",
    "Jayyid Jiddan",
    "Jayyid",
    "Maqbul",
    "Rosib"
]


LABEL_MAP = {
    label: idx
    for idx, label in enumerate(LABELS)
}

class EvaluationDataset(Dataset):

    def __init__(
        self,
        root_dir
    ):

        self.samples = []

        for label in LABELS:

            class_dir = os.path.join(
                root_dir,
                label
            )

            if not os.path.exists(
                class_dir
            ):
                continue

            files = glob.glob(
                os.path.join(
                    class_dir,
                    "*.wav"
                )
            )

            for file_path in files:

                self.samples.append(
                    (
                        file_path,
                        LABEL_MAP[label]
                    )
                )

        print(
            f"Evaluation data: "
            f"{len(self.samples)} audio"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        idx
    ):

        file_path, label = (
            self.samples[idx]
        )

        x = AudioUtil.preprocess(
            file_path,
            add_noise=False,
            shift=False
        )

        if x is None:

            x = torch.zeros(
                1,
                128,
                128
            )

        return (
            x,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

def create_model(
    device
):

    model = ContrastiveModel(
        mode="finetune"
    ).to(device)

    return model


@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    y_true = []
    y_pred = []

    for x, y in loader:

        x = x.to(device)

        output = model(x)

        pred = torch.argmax(
            output,
            dim=1
        )

        y_true.extend(
            y.numpy()
        )

        y_pred.extend(
            pred.cpu().numpy()
        )

    return (
        np.array(y_true),
        np.array(y_pred)
    )


if __name__ == "__main__":

    print("=" * 70)
    print("EVALUATION SSL")
    print("PSEUDO-LABELING")
    print("WITHOUT CONTRASTIVE LEARNING")
    print("=" * 70)

    device = Config.DEVICE

    print(
        f"Device: {device}"
    )

    dataset = EvaluationDataset(
        Config.LABELED_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    all_results = []

    for seed in Config.SEEDS:

        set_seed(seed)

        print()
        print("=" * 70)
        print(
            f"EVALUATION SEED {seed}"
        )
        print("=" * 70)

        model_path = os.path.join(
            SSL_MODEL_DIR,
            f"ssl_classifier_seed_{seed}.pth"
        )

        if not os.path.exists(
            model_path
        ):

            print(
                f"Model tidak ditemukan: "
                f"{model_path}"
            )

            continue

        model = create_model(
            device
        )

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        y_true, y_pred = evaluate(
            model,
            loader,
            device
        )

        accuracy = accuracy_score(
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

        print()
        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            zero_division=0
        )

        print()
        print(
            "Classification Report"
        )
        print(
            report
        )

        with open(
            os.path.join(
                SSL_RESULT_DIR,
                f"classification_report_seed_{seed}.txt"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            f.write(report)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS
        )

        disp.plot(
            ax=ax,
            values_format="d"
        )

        plt.title(
            f"SSL Confusion Matrix - Seed {seed}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                SSL_FIGURE_DIR,
                f"confusion_matrix_seed_{seed}.png"
            ),
            dpi=300
        )

        plt.close()

        all_results.append({
            "seed": seed,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    result_df = pd.DataFrame(
        all_results
    )

    if len(result_df) > 0:

        result_df.to_csv(
            os.path.join(
                SSL_RESULT_DIR,
                "ssl_results_all_seeds.csv"
            ),
            index=False
        )

        summary = pd.DataFrame({
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ],

            "mean": [
                result_df["accuracy"].mean(),
                result_df["precision"].mean(),
                result_df["recall"].mean(),
                result_df["f1_score"].mean()
            ],

            "std": [
                result_df["accuracy"].std(),
                result_df["precision"].std(),
                result_df["recall"].std(),
                result_df["f1_score"].std()
            ]
        })

        summary.to_csv(
            os.path.join(
                SSL_RESULT_DIR,
                "ssl_summary_mean_std.csv"
            ),
            index=False
        )

        print()
        print("=" * 70)
        print("SSL FINAL RESULT")
        print("=" * 70)

        print(
            result_df.to_string(
                index=False
            )
        )

        print()
        print(
            "Mean ± Std"
        )

        for _, row in summary.iterrows():

            print(
                f"{row['metric']:10s} : "
                f"{row['mean']:.4f} ± "
                f"{row['std']:.4f}"
            )

    else:

        print(
            "Tidak ada hasil evaluasi."
        )