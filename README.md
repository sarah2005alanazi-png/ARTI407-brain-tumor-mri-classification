# Brain Tumor Classification from MRI Images Using Deep Learning and Image Processing Techniques

ARTI 407 – Image Processing, Imam Abdulrahman bin Faisal University

**Team members:** Sarah Abdulrahman Al-Anzi (Leader), Haya Abdu Hakami, Danah Alhajri, Aseel AlQhtani, Rimas Alotaibi

## Overview
This project classifies brain MRI images into four classes (glioma, meningioma, pituitary tumor, no tumor). Images are first improved with classical image-processing techniques from the course (resize, median filter, CLAHE), then classified with three deep-learning models that are compared: a CNN trained from scratch, VGG16 and ResNet50 (transfer learning, with and without fine-tuning).

## Dataset
[Brain Tumor MRI Dataset (Kaggle, Version 2)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) – 7,200 images: 5,600 in `Training` and 1,600 in `Testing` (400 per class in the test set). The `Training` folder is split 80/20 (stratified) into training and validation. The `Testing` folder is used only for the final evaluation.

## Pipeline
1. Data exploration (class balance, image sizes, samples)
2. Preprocessing: resize to 224×224, 3×3 median filter, CLAHE
3. Data augmentation (flip, rotation, translation, zoom) on training images only
4. Models: CNN from scratch, VGG16, ResNet50 (frozen backbone, then fine-tuning the last block)
5. Evaluation on the test set: accuracy, macro precision / recall / F1, confusion matrices
6. Error analysis and Grad-CAM

## Results (test set, final run)
| Model | Accuracy | Macro F1 | Tumors missed as "no tumor" (of 1,200) |
|---|---|---|---|
| VGG16 (fine-tuned) | 0.9256 | 0.9237 | 39 |
| ResNet50 (fine-tuned) | 0.9250 | 0.9231 | 22 |
| CNN from scratch | 0.8812 | 0.8798 | 93 |
| ResNet50 (frozen) | 0.8681 | 0.8655 | 44 |
| VGG16 (frozen) | 0.8212 | 0.8143 | 67 |

Over three independent runs of the same code, the accuracy of the fine-tuned VGG16 and ResNet50 was 92.7 ± 0.4% and 91.9 ± 0.6%, so their difference is within run-to-run variation. Results change slightly between runs because GPU training is not fully deterministic. The full table is in `results/model_comparison.csv` and the figures are the `fig_*.png` files in the repository root.

## How to run
The notebook was run on **Kaggle** (GPU T4 x2, Internet on, to download the ImageNet weights) with the dataset above attached. Open `ARTI407_Brain_Tumor_MRI_Classification.ipynb`, attach the dataset, and run all cells from top to bottom (about 40 minutes). The dataset path in Step 1 (`/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset`) must be changed if the notebook is run outside Kaggle. Required packages are in `requirements.txt`.

## Limitations
- The test set comes from a different distribution than the training folder, which explains the gap between validation and test accuracy.
- Glioma is the hardest class (recall about 77%); it is mostly confused with meningioma and, in some T2/FLAIR slices, with "no tumor".
- The model may partly rely on image type and slice orientation rather than the lesion; Grad-CAM on a small sample suggests this but does not prove it.
