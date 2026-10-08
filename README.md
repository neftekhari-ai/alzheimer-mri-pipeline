# Alzheimer's MRI Staging Pipeline

A modular, two-stage deep learning pipeline for Alzheimer's disease staging from MRI scans, built with PyTorch.

**Stage 1 — Self-Supervised Pretraining (SimCLR):** learns visual representations from unlabeled MRI slices.
**Stage 2 — Supervised Fine-Tuning (ResNet-50):** classifies disease stage (CN / MCI / AD).
**Interpretability:** Grad-CAM heatmaps highlighting anatomically relevant regions (ventricles, hippocampus).

## Architecture

MRI Slices ──> SimCLR Pretraining (Stage 1)

│

▼

ResNet-50 Encoder (transfer)

│

▼

3-Class Classifier (Stage 2)

│

▼

Grad-CAM Interpretability Layer

                                                                    text

## Project Structure

alzheimer-mri-pipeline/

├── config.py # Central configuration (paths, hyperparameters)

├── train_simclr.py # Stage 1: self-supervised pretraining

├── train_classifier.py # Stage 2: supervised fine-tuning

├── inference.py # Prediction + Grad-CAM visualization

├── requirements.txt

└── src/

├── dataset.py # Data loading & preprocessing

├── model.py # ResNet-50 / SimCLR model definitions

├── simclr.py # SimCLR training logic (NT-Xent loss)

└── gradcam.py # Grad-CAM interpretability

                                                                    text

## Quick Start
```bash
pip install -r requirements.txt

# Stage 1: Self-supervised pretraining
python train_simclr.py

# Stage 2: Supervised classification
python train_classifier.py

# Inference with Grad-CAM
python inference.py --image <path_to_mri_slice>

Key Highlights

    Self-supervised learning (SimCLR) to exploit unlabeled medical imaging data
    Transfer learning with ResNet-50 for robust 3-class staging
    Grad-CAM interpretability — predictions aligned with clinically relevant anatomy
    Modular, production-ready design — clean separation of data, model, and training logic

Tech Stack

Python · PyTorch · Torchvision · OpenCV · Scikit-learn · NumPy

📬 Contact: neftekhari.ai@proton.me
