import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import glob
import csv
import json
import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

from src.config import Config, set_seed
from src.model import ContrastiveModel


DEVICE = Config.DEVICE

MODEL_DIR = os.path.join(
    Config.MODEL_DIR,
    "ssl"
)

RESULT_DIR = os.path.join(
    Config.RESULT_DIR,
    "ssl"
)

HISTORY_DIR = os.path.join(
    Config.HISTORY_DIR,
    "ssl"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


LABELS = [
    "MUMTAZ",
    "JAYYID_JIDDAN",
    "JAYYID",
    "MAQBUL",
    "RASIB"
]

LABEL_MAP = {
    "MUMTAZ": 0,
    "JAYYID_JIDDAN": 1,
    "JAYYID": 2,
    "MAQBUL": 3,
    "RASIB": 4
}

class SSLPrecomputedDataset(Dataset):

    def __init__(
        self,
        files,
        labels
    ):

        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:
            feature = torch.load(
                path,
                map_location="cpu"
            )

            if isinstance(feature, dict):

                if "feature" in feature:
                    feature = feature["feature"]

                elif "spectrogram" in feature:
                    feature = feature["spectrogram"]

                else:
                    feature = list(
                        feature.values()
                    )[0]

            feature = feature.float()

        except Exception as e:

            print(
                f"Gagal membaca: {path}"
            )
            print(e)

            feature = torch.zeros(
                1,
                64,
                64
            )

        label = self.labels[idx]

        return (
            feature,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


class UnlabeledDataset(Dataset):

    def __init__(
        self,
        files
    ):

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:

            feature = torch.load(
                path,
                map_location="cpu"
            )

            if isinstance(feature, dict):

                if "feature" in feature:
                    feature = feature["feature"]

                elif "spectrogram" in feature:
                    feature = feature["spectrogram"]

                else:
                    feature = list(
                        feature.values()
                    )[0]

            feature = feature.float()

        except Exception as e:

            print(
                f"Gagal membaca: {path}"
            )
            print(e)

            feature = torch.zeros(
                1,
                64,
                64
            )

        return (
            feature,
            path
        )


class PseudoLabeledDataset(Dataset):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        try:

            feature = torch.load(
                path,
                map_location="cpu"
            )

            if isinstance(feature, dict):

                if "feature" in feature:
                    feature = feature["feature"]

                elif "spectrogram" in feature:
                    feature = feature["spectrogram"]

                else:
                    feature = list(
                        feature.values()
                    )[0]

            feature = feature.float()

        except Exception as e:

            print(
                f"Gagal membaca: {path}"
            )
            print(e)

            feature = torch.zeros(
                1,
                64,
                64
            )

        return (
            feature,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

def load_split(
    seed
):

    split_path = os.path.join(
        Config.SPLIT_DIR,
        f"seed_{seed}.json"
    )

    if not os.path.exists(
        split_path
    ):

        raise FileNotFoundError(
            f"Split tidak ditemukan: "
            f"{split_path}"
        )

    with open(
        split_path,
        "r"
    ) as f:

        split = json.load(f)

    return split

def get_label_from_path(
    path
):

    filename = os.path.basename(
        path
    )

    for class_name, label in LABEL_MAP.items():

        if filename.startswith(
            class_name
        ):

            return label

    raise ValueError(
        f"Label tidak dikenali: {filename}"
    )

def create_train_dataset(
    split
):

    train_files = split["train"]

    train_labels = [
        get_label_from_path(
            path
        )
        for path in train_files
    ]

    dataset = SSLPrecomputedDataset(
        train_files,
        train_labels
    )

    return dataset

def create_validation_dataset(
    split
):

    validation_files = split[
        "validation"
    ]

    validation_labels = [
        get_label_from_path(
            path
        )
        for path in validation_files
    ]

    dataset = SSLPrecomputedDataset(
        validation_files,
        validation_labels
    )

    return dataset


def create_unlabeled_dataset():

    unlabeled_dir = (
        Config.PRECOMPUTED_UNLABELED
    )

    if not os.path.exists(
        unlabeled_dir
    ):

        raise FileNotFoundError(
            f"Folder unlabeled tidak ditemukan: "
            f"{unlabeled_dir}"
        )

    files = []

    for filename in os.listdir(
        unlabeled_dir
    ):

        if filename.endswith(
            ".pt"
        ):

            files.append(
                os.path.join(
                    unlabeled_dir,
                    filename
                )
            )

    files = sorted(
        files
    )

    print(
        f"Jumlah unlabeled: "
        f"{len(files)}"
    )

    return UnlabeledDataset(
        files
    )


def create_model():

    # ========================================================
    # PENTING:
    #
    # pretrained=True:
    # menggunakan ImageNet pretrained ResNet-18.
    #
    #
    # Jadi baseline ini:
    # SSL + Pseudo-Labeling
    # WITHOUT Contrastive Learning
    # ========================================================

    model = ContrastiveModel(
        num_classes=len(LABELS),
        mode="finetune",
        pretrained=True
    )

    model = model.to(
        DEVICE
    )

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_data = 0

    for inputs, targets in tqdm(
        loader,
        leave=False
    ):

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            inputs
        )

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_correct += (
            predictions == targets
        ).sum().item()

        total_data += (
            targets.size(0)
        )

    avg_loss = (
        total_loss /
        max(len(loader), 1)
    )

    accuracy = (
        100.0 *
        total_correct /
        max(total_data, 1)
    )

    return (
        avg_loss,
        accuracy
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_data = 0

    for inputs, targets in loader:

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            inputs
        )

        loss = criterion(
            outputs,
            targets
        )

        total_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_correct += (
            predictions == targets
        ).sum().item()

        total_data += (
            targets.size(0)
        )

    avg_loss = (
        total_loss /
        max(len(loader), 1)
    )

    accuracy = (
        100.0 *
        total_correct /
        max(total_data, 1)
    )

    return (
        avg_loss,
        accuracy
    )


def initial_training(
    model,
    train_loader,
    val_loader,
    seed
):

    print()
    print("=" * 70)
    print(
        f"INITIAL TRAINING | SEED {seed}"
    )
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE
    )

    best_acc = -1.0
    patience = 0

    history = []

    best_model_path = os.path.join(
        MODEL_DIR,
        f"ssl_initial_seed_{seed}.pth"
    )

    for epoch in range(
        Config.EPOCHS_FINETUNE
    ):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        scheduler.step(
            val_acc
        )

        history.append({
            "phase": "initial",
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Epoch [{epoch + 1}/"
            f"{Config.EPOCHS_FINETUNE}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:

            print(
                f"Best Val Acc: "
                f"{best_acc:.2f}% -> "
                f"{val_acc:.2f}%"
            )

            best_acc = val_acc
            patience = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

        else:

            patience += 1

        if (
            patience >=
            Config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping initial training."
            )

            break

    return (
        history,
        best_model_path
    )


@torch.no_grad()
def generate_pseudo_labels(
    model,
    unlabeled_dataset,
    seed,
    threshold=0.95
):

    print()
    print("=" * 70)
    print(
        f"PSEUDO-LABELING | SEED {seed}"
    )
    print(
        f"Confidence Threshold: "
        f"{threshold}"
    )
    print("=" * 70)

    loader = DataLoader(
        unlabeled_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    model.eval()

    pseudo_samples = []

    confidence_values = []

    for inputs, paths in tqdm(
        loader,
        desc="Generating pseudo labels"
    ):

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            inputs
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predictions = (
            torch.max(
                probabilities,
                dim=1
            )
        )

        for i in range(
            len(paths)
        ):

            conf = confidence[
                i
            ].item()

            prediction = predictions[
                i
            ].item()

            confidence_values.append(
                conf
            )

            if conf >= threshold:

                pseudo_samples.append(
                    (
                        paths[i],
                        prediction
                    )
                )

    pseudo_rows = []

    for path, label in pseudo_samples:

        pseudo_rows.append({
            "file_path": path,
            "pseudo_label": label,
            "class_name": LABELS[label]
        })

    pseudo_df = pd.DataFrame(
        pseudo_rows
    )

    pseudo_path = os.path.join(
        RESULT_DIR,
        f"pseudo_labels_seed_{seed}.csv"
    )

    pseudo_df.to_csv(
        pseudo_path,
        index=False
    )

    total_unlabeled = len(
        unlabeled_dataset
    )

    total_pseudo = len(
        pseudo_samples
    )

    pseudo_ratio = (
        100.0 *
        total_pseudo /
        max(total_unlabeled, 1)
    )

    mean_confidence = (
        sum(confidence_values) /
        max(
            len(confidence_values),
            1
        )
    )

    print()
    print(
        f"Total unlabeled       : "
        f"{total_unlabeled}"
    )

    print(
        f"Pseudo-labeled        : "
        f"{total_pseudo}"
    )

    print(
        f"Terpilih              : "
        f"{pseudo_ratio:.2f}%"
    )

    print(
        f"Mean confidence       : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"File pseudo-label     : "
        f"{pseudo_path}"
    )

    return pseudo_samples


def ssl_training(
    model,
    combined_dataset,
    val_loader,
    seed
):

    print()
    print("=" * 70)
    print(
        f"SSL TRAINING | SEED {seed}"
    )
    print("=" * 70)

    loader = DataLoader(
        combined_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE
    )

    best_acc = -1.0
    patience = 0

    history = []

    final_model_path = os.path.join(
        MODEL_DIR,
        f"ssl_classifier_seed_{seed}.pth"
    )

    for epoch in range(
        Config.EPOCHS_FINETUNE
    ):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        scheduler.step(
            val_acc
        )

        history.append({
            "phase": "ssl",
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Epoch [{epoch + 1}/"
            f"{Config.EPOCHS_FINETUNE}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:

            print(
                f"Best Val Acc: "
                f"{best_acc:.2f}% -> "
                f"{val_acc:.2f}%"
            )

            best_acc = val_acc
            patience = 0

            torch.save(
                model.state_dict(),
                final_model_path
            )

        else:

            patience += 1

        if (
            patience >=
            Config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping SSL."
            )

            break

    return (
        history,
        final_model_path
    )


def train_one_seed(
    seed,
    threshold=0.95
):

    print()
    print("#" * 70)
    print(
        f"SEMI-SUPERVISED LEARNING | SEED {seed}"
    )
    print(
        "WITHOUT CONTRASTIVE LEARNING"
    )
    print("#" * 70)

    set_seed(seed)

    split = load_split(
        seed
    )

    print(
        f"Train      : "
        f"{len(split['train'])}"
    )

    print(
        f"Validation : "
        f"{len(split['validation'])}"
    )

    print(
        f"Test       : "
        f"{len(split['test'])}"
    )

    train_dataset = create_train_dataset(
        split
    )

    val_dataset = create_validation_dataset(
        split
    )

    unlabeled_dataset = (
        create_unlabeled_dataset()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    model = create_model()

    print()
    print(
        "Model SSL dibuat dari:"
    )

    print(
        "ResNet-18 ImageNet pretrained"
    )


    initial_history, initial_model_path = (
        initial_training(
            model,
            train_loader,
            val_loader,
            seed
        )
    )

    model.load_state_dict(
        torch.load(
            initial_model_path,
            map_location=DEVICE
        )
    )

    pseudo_samples = (
        generate_pseudo_labels(
            model,
            unlabeled_dataset,
            seed,
            threshold
        )
    )

    pseudo_dataset = (
        PseudoLabeledDataset(
            pseudo_samples
        )
    )

    combined_dataset = ConcatDataset([
        train_dataset,
        pseudo_dataset
    ])

    print()
    print("=" * 70)
    print("DATASET SSL")
    print("=" * 70)

    print(
        f"Labeled train     : "
        f"{len(train_dataset)}"
    )

    print(
        f"Pseudo labeled    : "
        f"{len(pseudo_dataset)}"
    )

    print(
        f"Total training    : "
        f"{len(combined_dataset)}"
    )

    ssl_history, final_model_path = (
        ssl_training(
            model,
            combined_dataset,
            val_loader,
            seed
        )
    )

    history = (
        initial_history +
        ssl_history
    )

    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(
        HISTORY_DIR,
        f"ssl_history_seed_{seed}.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print()
    print(
        f"History disimpan: "
        f"{history_path}"
    )

    print(
        f"Model disimpan: "
        f"{final_model_path}"
    )

    return {
        "seed": seed,
        "pseudo_count": len(
            pseudo_samples
        ),
        "final_model": final_model_path,
        "history": history_path
    }


if __name__ == "__main__":

    print("=" * 70)
    print(
        "SEMI-SUPERVISED LEARNING"
    )
    print(
        "PSEUDO-LABELING"
    )
    print("=" * 70)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Seeds  : {Config.SEEDS}"
    )

    print(
        "Threshold : 0.95"
    )

    results = []

    for seed in Config.SEEDS:

        try:

            result = train_one_seed(
                seed,
                threshold=0.95
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"[ERROR] Seed {seed}"
            )

            print(e)

    print()
    print("=" * 70)
    print(
        "SEMUA TRAINING SSL SELESAI"
    )
    print("=" * 70)

    print(
        f"Berhasil: "
        f"{len(results)} / "
        f"{len(Config.SEEDS)} seed"
    )