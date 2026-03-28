# Bocconi Nuclei Challenge

Bocconi University — Computer Vision and Image Processing  
Competition: [Kaggle — MoNuSeg 2026](https://kaggle.com/competitions/mo-nu-seg-2026-cell-nucleus-detection)

Two-task Kaggle competition on nucleus detection and semantic segmentation in H&E-stained histopathology patches, derived from the [MoNuSeg](https://monuseg.grand-challenge.org/) dataset.

---

## Results

| Task | Place | Kaggle test score | Metric | Val score |
|---|---|---|---|---|
| Detection | **1st / ~40** | **0.7957** | F1 @ IoU 0.5 | 0.8296 |
| Segmentation | **2nd / ~40** | **0.8361** | Mean Dice | 0.8389 |

Kaggle test scores are from the held-out leaderboard evaluation.  
Val scores are from a slide-level held-out split (see [Design decisions](#design-decisions)).  
The close val/test gap on segmentation (0.8389 → 0.8361) confirms the slide-level split was not overfit.

---

## Repository structure

```
.
├── detection_notebook.ipynb      # Faster R-CNN fine-tuning + detection submission
├── segmentation_notebook.ipynb   # U-Net fine-tuning + TTA + segmentation submission
└── README.md
```

Both notebooks run on Google Colab (T4 GPU). See [Setup](#setup).

---

## Task 1 — Detection (1st place, F1@IoU0.5: 0.7957)

**Goal:** Predict bounding boxes around individual nuclei. One submission row per image; boxes serialized as `x1 y1 x2 y2; ...`  
**Metric:** Global F1 at IoU = 0.5. A predicted box is a true positive only if it matches an unmatched ground-truth box at IoU ≥ 0.5.

### Approach

| Decision | Starter baseline | This solution |
|---|---|---|
| Backbone | MobileNet V3 | ResNet-50 FPN v2 (pretrained COCO) |
| Training data | 35% | 100% |
| Augmentation | None | H-flip, V-flip, ColorJitter (bbox-aware) |
| Epochs | 3 | 12 |
| LR schedule | None | Cosine annealing |
| Score threshold | Fixed 0.50 | Swept on validation → 0.60 |

**Key implementation details:**

- Only the box predictor head is replaced; the FPN backbone keeps COCO pretrained weights
- Spatial augmentations (flips) correctly update bounding box coordinates — color jitter does not require coordinate updates
- Score threshold is swept across the validation set and fixed at 0.60 for test submission
- Custom F1 evaluation with per-image IoU matching, no third-party eval dependency

---

## Task 2 — Segmentation (2nd place, Dice: 0.8361)

**Goal:** Predict one binary pixel mask per image (nucleus pixels = 1, background = 0) in RLE format.  
**Metric:** Mean Dice score over the test set.

Two segmentation approaches are in this repo: an initial from-scratch U-Net in `detection_notebook.ipynb` (used for the joint submission) and the improved dedicated notebook. The dedicated notebook is the final submitted solution for the segmentation leaderboard.

### Approach (`segmentation_notebook.ipynb`)

| Decision | Initial approach | Final approach |
|---|---|---|
| Encoder | Scratch U-Net (no BN) | ResNet-34 pretrained on ImageNet |
| Library | Custom PyTorch | segmentation-models-pytorch |
| Augmentation | H-flip, V-flip, ColorJitter | + ElasticTransform, GridDistortion, CLAHE, stain simulation |
| Loss | 0.6 BCE + 0.4 Dice | 0.5 BCE + 0.5 Dice |
| LR schedule | Cosine | OneCycleLR (10% warmup → cosine decay) |
| Epochs | 20 | 30 with best-checkpoint saving |
| Inference | Single pass | 3-pass TTA (original + H-flip + V-flip) |
| Threshold | Fixed 0.35 | Grid-searched on validation → 0.40 |

**Key implementation details:**

- ImageNet mean/std normalization applied because the encoder is ImageNet-pretrained
- CLAHE improves contrast in H&E patches specifically — standard photometric augmentation is less effective here
- ElasticTransform and GridDistortion simulate realistic tissue deformation
- HueSaturationValue with narrow hue range approximates stain color variation across slide batches
- TTA averages sigmoid probability maps (not binarized masks) before thresholding
- Best checkpoint saved during training by val Dice; reloaded before inference and threshold search

---

## Design decisions

**Slide-level validation split.** The dataset contains 256×256 patches cropped from a smaller set of whole-slide images. A random image-level split leaks: patches from the same slide appear in both train and val, inflating validation metrics because the model has seen adjacent tissue. Splitting by source slide ensures validation images come from tissue withheld entirely. The tight val/test gap on segmentation (0.8389 → 0.8361) is evidence that this split was clean.

**TTA on histology patches.** H&E patches have no canonical orientation — a nucleus looks identical flipped or rotated. Averaging predictions over the original image, a horizontal flip, and a vertical flip (with prediction maps flipped back before averaging) reduces variance at near-zero cost. Applied after loading the best checkpoint, not during training.

**Threshold tuning on both tasks.** The detection score threshold and the segmentation probability threshold are both swept on the validation set rather than left at defaults. For detection, the default 0.50 over-suppresses low-confidence true positives in dense nucleus fields. For segmentation, the optimal threshold varies with class imbalance and loss weighting. Both sweeps are cheap and reliably improve the submitted metric.

---

## Setup

### Dataset

The competition dataset is derived from [MoNuSeg](https://monuseg.grand-challenge.org/) (Multi-Organ Nucleus Segmentation), a publicly available histopathology benchmark. The Kaggle packaging adds detection labels (bounding boxes derived from instance masks) and formats submissions per-image rather than per-nucleus.

### Requirements

- Python 3.10+
- GPU with ≥ 8GB VRAM (tested on Colab T4)

```bash
pip install torch torchvision segmentation-models-pytorch albumentations \
            tqdm pandas numpy pillow opencv-python-headless
```

### Running on Colab

Both notebooks mount Google Drive and read the dataset from a zip. Update `BASE_DIR` and `ZIP_PATH` at the top of each notebook to match your Drive layout, then run all cells in order.
