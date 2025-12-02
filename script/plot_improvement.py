import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. OVERALL TEST ACCURACY – THREE STAGES
#    Baseline → CB-CE → ImageNet+CB-CE
# ============================================================

stages = ["Baseline\nResNet18",
          "ResNet18\n+ CB-CE",
          "ResNet18 (ImageNet)\n+ CB-CE"]

overall_acc = np.array([
    0.6885,  # Baseline ResNet18
    0.7024,  # ResNet18 + CB-CE
    0.8022   # ResNet18 (ImageNet pretrained) + CB-CE
])

plt.figure(figsize=(6, 4))
x = np.arange(len(stages))

# Line + markers to show progression
plt.plot(x, overall_acc, marker="o", linewidth=2.5)
plt.xticks(x, stages)
plt.ylim(0.5, 0.9)
plt.ylabel("Test Accuracy")
plt.title("Overall Test Accuracy Across Three Stages")

for i, acc in enumerate(overall_acc):
    plt.text(x[i], acc + 0.01, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("overall_accuracy_three_stages.png", dpi=300)
plt.close()


# ============================================================
# 2. PER-CLASS TEST ACCURACY – THREE STAGES
#    Each class is a colored line; DF is emphasized
# ============================================================

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# ---- Per-class accuracies from your runs ----
# Baseline ResNet18
acc_base = np.array([
    0.4971,  # MEL
    0.8889,  # NV
    0.4516,  # BCC
    0.2093,  # AKIEC
    0.3594,  # BKL
    0.0000,  # DF
    0.5429,  # VASC
])

# ResNet18 + CB-CE
acc_cbce = np.array([
    0.4211,  # MEL
    0.9142,  # NV
    0.4409,  # BCC
    0.2093,  # AKIEC
    0.3917,  # BKL
    0.0000,  # DF
    0.6857,  # VASC
])

# ResNet18 (ImageNet) + CB-CE – final model
acc_final = np.array([
    0.4327,  # MEL
    0.9318,  # NV
    0.8065,  # BCC
    0.5814,  # AKIEC
    0.6406,  # BKL
    0.6591,  # DF
    0.6857,  # VASC
])

per_class_acc = {
    "Baseline ResNet18": acc_base,
    "ResNet18 + CB-CE": acc_cbce,
    "ResNet18 (ImageNet) + CB-CE": acc_final,
}

stage_labels = ["Baseline", "CB-CE", "ImageNet+CB-CE"]
x_stages = np.arange(len(stage_labels))

plt.figure(figsize=(8, 5))

# Optional: use a colormap to get distinct colors
cmap = plt.get_cmap("tab10")

for i, cls in enumerate(classes):
    # Collect that class's accuracies across stages
    cls_acc = np.array([acc_base[i], acc_cbce[i], acc_final[i]])

    if cls == "DF":
        # Emphasize DF: thicker line, larger markers, label note
        plt.plot(
            x_stages,
            cls_acc,
            marker="o",
            linewidth=3.0,
            markersize=8,
            color="black",
            label=f"{cls} (DF - emphasized)",
        )
    else:
        plt.plot(
            x_stages,
            cls_acc,
            marker="o",
            linewidth=1.8,
            color=cmap(i % 10),
            label=cls,
        )

plt.xticks(x_stages, stage_labels)
plt.ylim(0.0, 1.0)
plt.ylabel("Per-Class Test Accuracy")
plt.title("Per-Class Test Accuracy Across Model Improvements")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("per_class_accuracy_three_stages.png", dpi=300)
plt.close()

print("Saved plots:")
print(" - overall_accuracy_three_stages.png")
print(" - per_class_accuracy_three_stages.png")
