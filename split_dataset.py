import os
import glob
import json
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config


INPUT_DIR = Config.PRECOMPUTED_LABELED
OUTPUT_DIR = Config.SPLIT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("INPUT DIR :", INPUT_DIR)
print("OUTPUT DIR:", OUTPUT_DIR)
print("=" * 60)

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(
        f"Folder tidak ditemukan:\n{INPUT_DIR}"
    )


files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.pt")
    )
)

if len(files) == 0:
    raise ValueError(
        "Tidak ditemukan file .pt pada folder precomputed_labeled."
    )

labels = []

LABELS = [
    "JAYYID_JIDDAN",
    "MUMTAZ",
    "JAYYID",
    "MAQBUL",
    "RASIB"
]

for file in files:
    filename = os.path.basename(file)
    label = None
    for cls in LABELS:
        if filename.startswith(cls):
            label = cls
            break
    if label is None:
        raise ValueError(
            f"Label tidak dikenali : {filename}"
        )
    labels.append(label)

print("=" * 60)
print("Total Data :", len(files))
print("=" * 60)

print("\nDistribusi Label Awal")

for label, total in Counter(labels).items():
    print(f"{label:15s}: {total}")


for seed in Config.SEEDS:

    print("\n" + "=" * 60)
    print(f"Seed : {seed}")
    print("=" * 60)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - Config.TRAIN_RATIO,
        random_state=seed
    )

    train_idx, remain_idx = next(
        splitter.split(files, labels)
    )

    train_files = [files[i] for i in train_idx]
    remain_files = [files[i] for i in remain_idx]
    remain_labels = [labels[i] for i in remain_idx]

    val_ratio = (
        Config.VAL_RATIO /
        (Config.VAL_RATIO + Config.TEST_RATIO)
    )

    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_ratio,
        random_state=seed
    )

    val_idx, test_idx = next(
        splitter2.split(
            remain_files,
            remain_labels
        )
    )

    validation_files = [
        remain_files[i]
        for i in val_idx
    ]

    test_files = [
        remain_files[i]
        for i in test_idx
    ]

    split = {
        "train": train_files,
        "validation": validation_files,
        "test": test_files
    }

    save_path = os.path.join(
        OUTPUT_DIR,
        f"seed_{seed}.json"
    )

    with open(save_path, "w") as f:

        json.dump(
            split,
            f,
            indent=4
        )

    print(f"Train      : {len(train_files)}")
    print(f"Validation : {len(validation_files)}")
    print(f"Test       : {len(test_files)}")

print("\n" + "=" * 60)
print("Semua file split berhasil dibuat.")
print("Disimpan pada :", OUTPUT_DIR)
print("=" * 60)