import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18

class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2018 Task 3 (HAM10000) images.
    """

    def __init__(self, df, image_dir, class_cols, transform=None):
        """
        df: DataFrame with column 'image_id' and one-hot class columns.
        image_dir: folder with all ISIC_*.jpg files.
        class_cols: list of diagnosis columns in df.
        transform: torchvision transforms to apply to each image.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.class_cols = class_cols
        self.transform = transform

        # convert one-hot labels to integer indices 0..C-1
        self.labels = np.argmax(self.df[self.class_cols].values, axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.image_dir, image_id + ".jpg")

        # load RGB image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


def load_metadata(csv_path):
    """
    Load the ground-truth CSV and return (df, class_cols).

    csv_path: path to training_ground_truth.csv
    """
    df = pd.read_csv(csv_path)

    # some versions use 'image' instead of 'image_id'
    if 'image' in df.columns and 'image_id' not in df.columns:
        df = df.rename(columns={'image': 'image_id'})

    class_cols = [c for c in df.columns if c != 'image_id']
    return df, class_cols


def split_dataset(df, class_cols, val_split=0.15, test_split=0.15, seed=42):
    """
    Split dataframe into train / val / test with stratification by class.
    Returns df_train, df_val, df_test, train_labels (integer indices).
    """
    labels_idx = np.argmax(df[class_cols].values, axis=1)

    # first split into train and temp
    df_train, df_temp, y_train, y_temp = train_test_split(
        df,
        labels_idx,
        test_size=val_split + test_split,
        random_state=seed,
        stratify=labels_idx,
    )

    # then split temp into val and test
    val_ratio = val_split / (val_split + test_split)
    df_val, df_test, _, _ = train_test_split(
        df_temp,
        y_temp,
        test_size=1 - val_ratio,
        random_state=seed,
        stratify=y_temp,
    )

    return df_train, df_val, df_test, y_train


def get_transforms(img_size=224):
    """
    Return (train_transform, eval_transform).
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def build_dataloaders(
    image_dir,
    csv_path,
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    img_size=224,
    seed=42,
    num_workers=4,
):
    """
    High-level helper that builds train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader,
        num_classes, class_cols, train_labels (numpy array of ints)
    """
    df, class_cols = load_metadata(csv_path)
    num_classes = len(class_cols)

    df_train, df_val, df_test, train_labels = split_dataset(
        df, class_cols, val_split=val_split, test_split=test_split, seed=seed
    )

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_ds = ISICDataset(df_train, image_dir, class_cols, transform=train_tf)
    val_ds = ISICDataset(df_val, image_dir, class_cols, transform=eval_tf)
    test_ds = ISICDataset(df_test, image_dir, class_cols, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, num_classes, class_cols, train_labels


def compute_class_weights(labels, num_classes):
    """
    Compute class weights (inverse frequency) for imbalanced data.

    labels: 1D array of integer class indices.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()  # normalize to ~1 on average

    print("Class counts:", counts)
    print("Class weights:", weights)

    return torch.tensor(weights, dtype=torch.float32)

def hybrid_sampling(df_train, class_cols, target_major=4000, target_minor=2000, seed=42):
    """
    Hybrid sampling for HAM10000-like data.

    - Mildly undersample the majority class (NV) down to target_major
    - Oversample minority classes up to target_minor
    - Keep medium-size classes as-is if they are between these thresholds

    Assumes:
      - df_train has one-hot columns in class_cols
      - class index 1 corresponds to NV in order:
        ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    """
    df = df_train.copy()
    # convert one-hot to integer labels
    label_idx = np.argmax(df[class_cols].values, axis=1)
    df["label_idx"] = label_idx

    # original label counts (0..6)
    class_counts = df["label_idx"].value_counts().sort_index()
    rng = np.random.default_rng(seed)

    parts = []

    for cls, count in class_counts.items():
        df_cls = df[df["label_idx"] == cls]

        # NV is class index 1
        if cls == 1 and count > target_major:
            # mild undersampling of NV
            keep_idx = rng.choice(df_cls.index, size=target_major, replace=False)
            parts.append(df.loc[keep_idx])
        elif count < target_minor:
            # oversample minority classes
            extra_idx = rng.choice(df_cls.index, size=target_minor - count, replace=True)
            df_extra = df.loc[extra_idx]
            parts.append(pd.concat([df_cls, df_extra], axis=0))
        else:
            # keep this class as-is
            parts.append(df_cls)

    df_balanced = pd.concat(parts, axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=["label_idx"])

    return df_balanced, class_counts

from sklearn.metrics import confusion_matrix

def evaluate_per_class_accuracy(model, loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Confusion matrix: rows = true, cols = predicted
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    print("\n================ Per-Class Accuracy ================\n")
    per_class_acc = {}

    for i, cls in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0.0
        per_class_acc[cls] = acc
        print(f"{cls:10s}: {acc:.4f}   ({correct}/{total})")

    print("\n====================================================\n")
    return per_class_acc

def create_resnet18_imagenet(num_classes, device):
    """
    Load ResNet-18 with ImageNet weights from a local checkpoint.
    If the file is missing, fall back to training from scratch.
    """
    if os.path.exists(RESNET18_PRETRAINED):
        print(f"\nLoading ImageNet ResNet-18 weights from {RESNET18_PRETRAINED}")
        # Load the checkpoint
        state_dict = torch.load(RESNET18_PRETRAINED, map_location=device)

        # Start from an uninitialized ResNet-18 and load all weights
        model = resnet18(weights=None)
        model.load_state_dict(state_dict)  # this matches the 1000-class head

        # Replace the final layer to match your 7 classes (randomly initialized)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.to(device)
    else:
        print(
            "\n[WARNING] Pretrained file not found at\n"
            f"  {RESNET18_PRETRAINED}\n"
            "Training ResNet-18 from scratch instead."
        )
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.to(device)

    return model

def build_fixed_dataloaders(
    train_csv, train_dir,
    val_csv, val_dir,
    test_csv, test_dir,
    batch_size=32,
    img_size=224,
):
    # Training metadata (defines the label columns)
    df_train, class_cols = load_metadata(train_csv)
    num_classes = len(class_cols)

    # Validation & test metadata (reuse the same class columns)
    df_val, _ = load_metadata(val_csv)
    df_test, _ = load_metadata(test_csv)

    # Transforms
    train_tf, eval_tf = get_transforms(img_size=img_size)

    # Datasets
    train_ds = ISICDataset(df_train, train_dir, class_cols, transform=train_tf)
    val_ds   = ISICDataset(df_val,   val_dir,   class_cols, transform=eval_tf)
    test_ds  = ISICDataset(df_test,  test_dir,  class_cols, transform=eval_tf)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    # For class-balanced weights, get integer labels from training df
    train_onehot = df_train[class_cols].values  # shape [N, num_classes]
    train_labels = np.argmax(train_onehot, axis=1).astype(np.int64)

    return train_loader, val_loader, test_loader, num_classes, class_cols, train_labels