# SkinVisionML

**Addressing Extreme Class Imbalance in Dermoscopic Skin Lesion Classification via Class-Balanced Loss and Two-Stage Fine-Tuning**

A deep learning project for **multiclass skin lesion classification** on the **HAM10000** dermoscopic image dataset, with a specific focus on improving performance for **minority and clinically important lesion classes** under severe class imbalance.

---

## Project Overview

Automated skin lesion classification can support earlier detection of malignant skin cancers, but real-world medical image datasets are often highly imbalanced. In this project, I developed a multiclass skin lesion classification pipeline that improves minority-class recognition by combining:

- a **ResNet-18** backbone,
- **class-balanced cross-entropy loss**,
- **ImageNet-based transfer learning**, and
- a **two-stage fine-tuning strategy**.

Rather than introducing a new architecture, this project focuses on improving **training dynamics, optimization stability, and class-wise performance** in an imbalanced medical imaging setting.

---

## Key Highlights

- **Dataset:** HAM10000 dermoscopic lesion dataset
- **Task:** 7-class skin lesion classification
- **Backbone:** ResNet-18
- **Core methods:** class-balanced loss, transfer learning, staged fine-tuning
- **Main goal:** improve recognition of underrepresented lesion categories
- **Final test accuracy:** **80.22%**
- **Baseline test accuracy:** **68.85%**

The final model delivered substantial gains for minority classes such as **BCC, AKIEC, and DF**, while maintaining strong performance on the dominant **NV** class.

---

## Lesion Classes

The model is trained to classify dermoscopic images into the following seven diagnostic categories:

- **MEL** — Melanoma
- **NV** — Melanocytic nevi
- **BCC** — Basal cell carcinoma
- **AKIEC** — Actinic keratoses / intraepithelial carcinoma
- **BKL** — Benign keratosis-like lesions
- **DF** — Dermatofibroma
- **VASC** — Vascular lesions

---

## Methodology

### 1. Baseline Model
The initial benchmark uses **ResNet-18** trained with standard cross-entropy loss.

### 2. Class-Balanced Loss
To reduce imbalance-driven failure on rare classes, the model uses **class-balanced cross-entropy**, which reweights classes based on the effective number of samples.

### 3. Transfer Learning
The ResNet-18 backbone is initialized with **ImageNet-pretrained weights** to provide stronger feature representations before domain adaptation.

### 4. Two-Stage Fine-Tuning
Training is performed in two stages:

- **Stage 1 (warm-up):** freeze the backbone and train only the final classifier layer
- **Stage 2 (fine-tuning):** unfreeze all layers and optimize the full network end-to-end

This staged setup improves optimization stability and helps the model adapt pretrained features to dermoscopic imagery more effectively.

---

## Results

### Overall Accuracy

| Model / Stage | Test Accuracy |
|---|---:|
| ResNet-18 baseline | 0.6885 |
| ResNet-18 + CB-CE | 0.7024 |
| ResNet-18 + ImageNet pretraining + CB-CE | **0.8022** |

### Per-Class Accuracy: Baseline vs Final Model

| Class | Baseline | Final | Change |
|---|---:|---:|---:|
| MEL | 0.497 | 0.433 | -6.4% |
| NV | 0.889 | 0.932 | +4.3% |
| BCC | 0.452 | 0.807 | +35.5% |
| AKIEC | 0.209 | 0.581 | +37.2% |
| BKL | 0.359 | 0.641 | +28.1% |
| DF | 0.000 | 0.659 | +65.9% |
| VASC | 0.543 | 0.686 | +14.3% |

These results show that the final pipeline improves performance primarily through **stronger minority-class discrimination**, rather than only boosting overall accuracy through majority-class gains.

---

## Repository Structure

```text
SkinVisionML/
├── output/
│   ├── models/
│   │   └── skin_cancer_classifier.pth
│   ├── sample_images/
│   │   ├── class-AKIEC/
│   │   ├── class-BCC/
│   │   ├── class-BKL/
│   │   ├── class-MEL/
│   │   ├── class-NV/
│   │   ├── class-VASC/
│   │   └── sample_image_metadata.csv
│   ├── test_samples/
│   ├── ham10000_report.txt
│   ├── pixel_histogram.png
│   └── sample_images.png
├── script/
│   ├── export_sample_image.py
│   ├── plot_improvement.py
│   ├── quality_check.py
│   ├── resnet18_cbce_imagenet.py
│   ├── train_ResNet18_cbce.py
│   ├── train_baseline.py
│   ├── train_resnet18_imagenet_cbce.py
│   └── utils.py
├── SkinCancerModel_report.pdf
└── .gitignore
```

---

## What This Repository Contains

- **Training scripts** for baseline and improved model pipelines
- **Model weights** for the final trained classifier
- **Sample images and output artifacts** for qualitative inspection
- **Plots and reports** summarizing dataset characteristics and model improvement
- **Project report** documenting motivation, method, experiments, and conclusions

---

## Reproducing the Project

Because local paths and environment settings may differ, you should first review the scripts in the `script/` directory and update dataset/model paths as needed.

A typical workflow is:

1. Prepare the **HAM10000** dataset locally
2. Inspect and update paths in the training scripts
3. Train a baseline model using `train_baseline.py`
4. Train the class-balanced model using `train_ResNet18_cbce.py`
5. Train the final transfer-learning model using `train_resnet18_imagenet_cbce.py`
6. Review saved weights, plots, and qualitative outputs under `output/`

---

## Clinical / Research Note

This project is intended for **research and educational purposes only**. It is **not a clinical diagnostic tool** and should not be used for medical decision-making without proper clinical validation and regulatory review.

---

## Future Improvements

Potential next steps include:

- stronger data augmentation for rare lesion classes,
- synthetic minority-sample generation,
- external validation on additional dermoscopic datasets,
- self-supervised or domain-specific pretraining,
- improved calibration and interpretability analysis.

---

## Report

A full write-up of the project is included in:

- `SkinCancerModel_report.pdf`

The report covers the dataset, methodological design, baseline comparisons, per-class evaluation, qualitative analysis, discussion, and future work.

---

## Author

**Fiona Zhang**

If you use or reference this project, please cite the accompanying report or link back to this repository.
