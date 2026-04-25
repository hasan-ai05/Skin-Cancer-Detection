# Skin Cancer Detection + Grad-CAM

EfficientNet-B0 fine-tuned on 10,015 dermatoscopic images to classify 7 skin lesion types.
Grad-CAM visualizes exactly which region of each image drove the model's decision —
the heatmap highlights lesion boundaries, matching the patterns dermatologists use clinically.

---

## The Problem

Skin cancer is diagnosed through visual inspection of dermoscopy images.
The 7 lesion types are visually similar — melanoma and a benign mole can look nearly
identical to an untrained eye. A model that classifies but cannot explain its reasoning
is of limited clinical value.

This project addresses both: **what is the lesion type, and where in the image did
the model find evidence for that decision?**

---

## Dataset

**Source:** HAM10000 — Human Against Machine with 10,000 Training Images (ISIC 2018 Challenge)  
**Records:** 10,015 dermatoscopic images from Austrian and Australian clinics  
**Task:** 7-class classification

| Class | Full Name | Count | % |
|---|---|---|---|
| nv | Melanocytic Nevi (moles) | 6,705 | 66.9% |
| mel | Melanoma | 1,113 | 11.1% |
| bkl | Benign Keratosis | 1,099 | 11.0% |
| bcc | Basal Cell Carcinoma | 514 | 5.1% |
| akiec | Actinic Keratoses | 327 | 3.3% |
| vasc | Vascular Lesions | 142 | 1.4% |
| df | Dermatofibroma | 115 | 1.1% |

**Diagnosis method:** All labels confirmed by histopathology (biopsy) — not visual estimation.

---

## Class Imbalance

nv = 67% of the dataset. Without correction, the model learns to predict nv for everything
and achieves 67% accuracy while missing every malignant case.

**Solution — weighted CrossEntropyLoss:**

| Class | Weight | Reason |
|---|---|---|
| df | 12.36 | Rarest class — highest penalty for misclassification |
| akiec | 4.37 | Second rarest |
| bcc | 2.78 | — |
| mel | 1.29 | Dangerous despite moderate weight |
| nv | 0.21 | Most common — lowest penalty |

---

## Model

**EfficientNet-B0** pretrained on ImageNet (1.2M images, 1000 classes).

Transfer learning rationale: 10,015 images is too small to train a CNN from scratch.
EfficientNet already knows edges, textures, and color patterns from ImageNet.
Fine-tuning teaches it the difference between 7 skin lesion types in 10 epochs
instead of the 100+ epochs a randomly initialized network would require.

**Architecture:** 4,016,515 parameters, all trainable. Input size 224×224.

---

## Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.96 | 43.9% | 1.34 | 45.7% |
| 5 | ~0.52 | ~75% | ~0.73 | ~70% |
| 10 | 0.28 | 97.0% | 0.75 | 77.0% |

**Split:** 70% Train / 15% Val / 15% Test — stratified to maintain class ratios.

**Mild overfitting after epoch 4:** train loss continues falling while val loss plateaus.
Cause: 10,015 images is small for 7 classes. Val accuracy of 77% is the realistic ceiling
without additional data. Early stopping at epoch 4–5 would reduce the gap.

![Training Curves](images/training_curves.png)

---

## Results

**Test Accuracy: 78.71%**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| akiec | 0.55 | 0.76 | 0.64 | 49 |
| bcc | 0.69 | 0.77 | 0.73 | 77 |
| bkl | 0.61 | 0.69 | 0.65 | 165 |
| df | 0.47 | 0.41 | 0.44 | 17 |
| mel | — | **0.713** | — | — |
| nv | high | — | — | — |

**Melanoma Recall: 71.3%** — the clinically critical metric.
Of 100 real melanoma cases, 71 are correctly identified.
29 are missed — the number that matters in a clinical screening context.

![Confusion Matrix](images/confusion_matrix.png)

---

## Grad-CAM

Grad-CAM computes gradients of the predicted class score with respect to
the last convolutional layer. Regions that most influenced the decision
receive high gradient weights and appear red in the heatmap.

**What the heatmaps show:** The model focuses on lesion boundaries and color
inhomogeneity — the same visual cues dermatologists use clinically. The heatmap
is not centered on the middle of the image; it follows the irregular edges
of the lesion, confirming the model learned the correct features.

![Grad-CAM Melanoma](images/gradcam_melanoma.png)
![Grad-CAM Correct](images/gradcam_correct.png)

---

## How to Run

```bash
pip install kagglehub timm torch torchvision pandas numpy plotly matplotlib
jupyter notebook Project4_Skin_Cancer_Detection.ipynb
```

Dataset downloads automatically via `kagglehub`. GPU recommended — training takes
approximately 15 minutes on a T4 GPU in Google Colab.

---

## Where to Place Screenshots

```
skin-cancer-detection/
    images/
        class_distribution.png    Section 4 — bar chart
        sample_images.png         Section 6 — one sample per class
        training_curves.png       Section 13 — loss and accuracy
        confusion_matrix.png      Section 15 — 7x7 heatmap
        gradcam_melanoma.png      Section 17 — melanoma samples
        gradcam_correct.png       Section 17 — correctly classified
        executive_dashboard.png   Section 18 — full dashboard
    README.md
    Project4_Skin_Cancer_Detection.ipynb
```

---

## Project Structure

```
Project4_Skin_Cancer_Detection.ipynb    main notebook (18 sections)
README.md                               this file
images/                                 screenshots from notebook output
```

---

*Part of an 8-project AI Engineering portfolio — Hasan Akhras*
