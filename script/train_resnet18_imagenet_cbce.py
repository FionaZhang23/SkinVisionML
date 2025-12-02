import os
import sys
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18

# ------------------ PATHS / IMPORT UTILS ------------------ #

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import (
    load_metadata,
    get_transforms,
    ISICDataset,
    evaluate_per_class_accuracy,
)

# Data dirs
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "training")
VAL_DIR    = os.path.join(DATA_DIR, "validation")
TEST_DIR   = os.path.join(DATA_DIR, "test")

TRAIN_CSV = os.path.join(TRAIN_DIR, "training_ground_truth.csv")
VAL_CSV   = os.path.join(VAL_DIR, "validation_ground_truth.csv")
TEST_CSV  = os.path.join(TEST_DIR, "test_ground_truth.csv")

# Local pretrained ResNet-18 checkpoint
# (this is the .pth file you downloaded into script/)
RESNET18_PRETRAINED = os.path.join(
    CURRENT_DIR, "resnet18-f37072fd.pth"
)

# Hyperparams
BATCH_SIZE  = 32
IMG_SIZE    = 224
NUM_WORKERS = 4

NUM_EPOCHS_WARMUP   = 3    # only fc layer
NUM_EPOCHS_FINETUNE = 7    # full model
LR_WARMUP           = 1e-3
LR_FINETUNE         = 1e-4


# --------------------------------------------------------------------
#  BUILD DATALOADERS FROM EXISTING TRAIN / VAL / TEST FOLDERS
# --------------------------------------------------------------------
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
    train_onehot = df_train[class_cols].values  # [N, C]
    train_labels = np.argmax(train_onehot, axis=1).astype(np.int64)

    return train_loader, val_loader, test_loader, num_classes, class_cols, train_labels


# --------------------------------------------------------------------
#   CLASS-BALANCED WEIGHTS (Cui et al. 2019)
# --------------------------------------------------------------------
def compute_cb_weights(train_labels, num_classes, beta=0.99):
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.mean(weights)

    print("\nClass counts:", counts)
    print("CB weights (beta=%.3f): %s" % (beta, np.round(weights, 4)))

    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------
#   RESNET18 WITH LOCAL IMAGENET WEIGHTS
# --------------------------------------------------------------------
def create_resnet18_imagenet(num_classes, device):
    """
    Load ResNet-18 with ImageNet weights from a local checkpoint.
    If the file is missing, fall back to training from scratch.
    """
    if os.path.exists(RESNET18_PRETRAINED):
        print(f"\nLoading ImageNet ResNet-18 weights from {RESNET18_PRETRAINED}")
        state_dict = torch.load(RESNET18_PRETRAINED, map_location=device)

        # Start from uninitialized ResNet-18 and load all weights
        model = resnet18(weights=None)
        model.load_state_dict(state_dict)  # 1000-class head

        # Replace the final layer for your 7 classes
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


# --------------------------------------------------------------------
#  GENERIC TRAINING LOOP
# --------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device,
                criterion, optimizer, num_epochs=5, start_epoch_idx=1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_num = start_epoch_idx + epoch
        print(f"\nEpoch {epoch_num}/{start_epoch_idx + num_epochs - 1}")
        print("-" * 20)

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            _, preds = torch.max(out, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += torch.sum(preds == y).item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  ({time.time() - t0:.1f}s)")

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = criterion(out, y)
                _, preds = torch.max(out, 1)

                val_loss += loss.item() * x.size(0)
                val_correct += torch.sum(preds == y).item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val  Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_val_acc


# --------------------------------------------------------------------
#  TEST ACCURACY
# --------------------------------------------------------------------
def evaluate_accuracy(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            _, preds = torch.max(out, 1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# --------------------------------------------------------------------
#   MAIN
# --------------------------------------------------------------------
def main():

    # DEVICE
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # DATALOADERS
    (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_cols,
        train_labels,
    ) = build_fixed_dataloaders(
        train_csv=TRAIN_CSV,
        train_dir=TRAIN_DIR,
        val_csv=VAL_CSV,
        val_dir=VAL_DIR,
        test_csv=TEST_CSV,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    print("Classes:", class_cols)
    print(
        f"Train batches: {len(train_loader)}, "
        f"Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    # CLASS-BALANCED WEIGHTS
    cb_weights = compute_cb_weights(train_labels, num_classes, beta=0.99).to(device)

    # MODEL (pretrained if .pth exists)
    model = create_resnet18_imagenet(num_classes, device)

    # ------------------------------------------------------------
    #  STAGE 1: FREEZE BACKBONE, TRAIN ONLY FC
    # ------------------------------------------------------------
    print("\n=== Stage 1: train classifier head only (frozen backbone) ===")
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=cb_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_WARMUP,
    )

    model, best_val_acc_stage1 = train_model(
        model, train_loader, val_loader, device,
        criterion, optimizer,
        num_epochs=NUM_EPOCHS_WARMUP,
        start_epoch_idx=1,
    )
    print(f"Best val acc after Stage 1: {best_val_acc_stage1:.4f}")

    # ------------------------------------------------------------
    #  STAGE 2: UNFREEZE ALL LAYERS, FINE-TUNE
    # ------------------------------------------------------------
    print("\n=== Stage 2: fine-tune all layers ===")
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=cb_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

    model, best_val_acc_stage2 = train_model(
        model, train_loader, val_loader, device,
        criterion, optimizer,
        num_epochs=NUM_EPOCHS_FINETUNE,
        start_epoch_idx=NUM_EPOCHS_WARMUP + 1,
    )
    print(f"Best val acc after Stage 2: {best_val_acc_stage2:.4f}")

    # TEST ACCURACY
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\n[ResNet18 (ImageNet local) + CB-CE] Test Acc: {test_acc:.4f}")

    # PER-CLASS ACCURACY
    print("\nPer-class test accuracy:")
    evaluate_per_class_accuracy(model, test_loader, device, class_cols)


if __name__ == "__main__":
    main()
