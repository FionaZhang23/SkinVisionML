
import os
import sys
import time
import copy

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# torchvision models
from torchvision.models import (
    resnet18, resnet34, resnet50,
    mobilenet_v2,
    efficientnet_b0, efficientnet_b1,
    vgg11, vgg16
)

# ---------------- PATHS & IMPORTS ----------------


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
sys.path.append(PROJECT_ROOT)

from utils import (
    load_metadata,
    get_transforms,
    ISICDataset,
    evaluate_per_class_accuracy,
    build_fixed_dataloaders
)



DATA_ROOT       = "/Users/fionazhang/PycharmProjects/SkinVisionML/data"

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "training")
VAL_IMAGE_DIR   = os.path.join(DATA_ROOT, "validation")
TEST_IMAGE_DIR  = os.path.join(DATA_ROOT, "test")

TRAIN_CSV = os.path.join(TRAIN_IMAGE_DIR, "training_ground_truth.csv")
VAL_CSV   = os.path.join(VAL_IMAGE_DIR,   "validation_ground_truth.csv")
TEST_CSV  = os.path.join(TEST_IMAGE_DIR,  "test_ground_truth.csv")

BATCH_SIZE  = 32
IMG_SIZE    = 224
NUM_EPOCHS  = 5
NUM_WORKERS = 4
LR          = 1e-4


# ---------------- TRAINING ----------------

def train_one_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # ----- TRAIN -----
        model.train()
        total, correct = 0, 0
        running_loss = 0.0
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
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  ({time.time()-t0:.1f}s)")

        # ----- VALIDATION -----
        model.eval()
        val_total, val_correct = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = criterion(out, y)
                _, preds = torch.max(out, 1)

                val_loss += loss.item() * x.size(0)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_val_acc


# ---------------- TEST EVAL ----------------

def evaluate_accuracy(model, loader, device):

    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            _, preds = torch.max(out, 1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


# ---------------- MAIN ----------------

def main():
    # ----- DEVICE -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, num_classes, class_cols = \
        build_fixed_dataloaders(
            train_image_dir=TRAIN_IMAGE_DIR,
            train_csv=TRAIN_CSV,
            val_image_dir=VAL_IMAGE_DIR,
            val_csv=VAL_CSV,
            test_image_dir=TEST_IMAGE_DIR,
            test_csv=TEST_CSV,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            num_workers=NUM_WORKERS,
        )

    print("Classes:", class_cols)
    print("Num classes:", num_classes)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    model_builders = {
        "resnet18":        lambda: resnet18(weights=None),
        "resnet34":        lambda: resnet34(weights=None),
        "resnet50":        lambda: resnet50(weights=None),
        "mobilenet_v2":    lambda: mobilenet_v2(weights=None),
        "efficientnet_b0": lambda: efficientnet_b0(weights=None),
        "efficientnet_b1": lambda: efficientnet_b1(weights=None),
        "vgg11":           lambda: vgg11(weights=None),
        "vgg16":           lambda: vgg16(weights=None),
    }

    results = []

    for name, builder in model_builders.items():
        print("\n" + "=" * 70)
        print(f"Training baseline model: {name}")
        print("=" * 70)

        model = builder()

        if hasattr(model, "fc"): #for resnet
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unknown classifier structure for: {name}")

        model = model.to(device)

        # training
        trained_model, best_val_acc = train_one_model(
            model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS
        )

        # test test dataset accuracy
        test_acc = evaluate_accuracy(trained_model, test_loader, device)
        print(f"\n[{name}] Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

        # per-class accuracy
        print(f"\nPer-class test accuracy for {name}:")
        per_class_acc = evaluate_per_class_accuracy(
            trained_model, test_loader, device, class_cols
        )

        results.append({
            "model": name,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "per_class_acc": per_class_acc,
        })

    # ----- SUMMARY TABLE -----
    print("\n" + "#" * 80)
    print("Baseline Model Comparison (Fixed train/val/test, Standard CE)")
    print("#" * 80)
    print(f"{'Model':<20} {'Best Val Acc':>12} {'Test Acc':>12}")
    for r in results:
        print(f"{r['model']:<20} {r['best_val_acc']:.4f}{r['test_acc']:>12.4f}")


if __name__ == "__main__":
    main()
