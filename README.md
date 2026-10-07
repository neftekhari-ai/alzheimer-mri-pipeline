# Multi-Stage Alzheimer’s Disease Detection Pipeline from MRI Scans

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A robust, multi-stage Computer Vision pipeline leveraging Self-Supervised Learning (SimCLR) and Deep Convolutional Networks (ResNet-50) for fine-grained Alzheimer's disease staging using MRI scans.

---

## 📌 Executive Summary

Traditional medical image classification often suffers from limited labeled data and high intra-class variability. This project implements a **two-stage training strategy** (slice-level SSL pretraining followed by patient-level fine-tuning) on structural MRI data, enabling robust feature extraction from key anatomical regions such as the **hippocampus** and **ventricular spaces**.

---

## 🧠 Pipeline Architecture

The end-to-end pipeline consists of five key modular stages:

[ Raw MRI Scans ]

│

▼

[ Stage 1: Data Preprocessing & Slice Extraction ]

│

▼

[ Stage 2: Self-Supervised Pretraining (SimCLR) ]

│

▼

[ Stage 3: Feature Representation & Transfer (ResNet-50) ]

│

▼

[ Stage 4: Patient-Level Multi-Class Fine-Tuning ]

│

▼

[ Stage 5: Evaluation & Interpretability (Grad-CAM) ]


1. **Preprocessing & Normalization:** Volume slicing, intensity normalization, skull-stripping artifact mitigation, and contrast adjustments.
2. **Self-Supervised Pretraining (SimCLR):** Learns invariant latent representations from unlabeled slice augmentations without manual annotation bias.
3. **Backbone Architecture:** ResNet-50 backbone adapted for medical imaging feature spaces.
4. **Patient-Level Aggregation & Staging:** Multi-class classification fine-tuned at the patient cohort level.
5. **Model Interpretability:** Visual validation via Grad-CAM to ensure the model focuses on clinically relevant biomarkers (hippocampal atrophy and ventricular enlargement).

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision & Image Processing:** OpenCV, Scikit-Image, PIL
* **Scientific Computing:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure
```text
├── data/
│   ├── raw/                 # Raw MRI volumes / metadata
│   └── processed/           # Extracted 2D slices & normalized arrays
├── models/
│   ├── simclr_ssl.py        # Self-supervised contrastive learning module
│   ├── resnet_backbone.py   # Modified ResNet-50 architecture
│   └── classifier.py        # Patient-level classification head
├── pipelines/
│   ├── preprocess.py        # Slicing and normalization pipeline
│   ├── train_ssl.py         # SSL pretraining routine
│   └── train_finetune.py    # Downstream fine-tuning routine
├── evaluation/
│   ├── metrics.py           # Macro-F1, Confusion Matrix, AUC
│   └── gradcam.py           # Attention map visualization
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

🚀 Quickstart
1. Clone the repository

                                                                    bash
git clone https://github.com/YOUR_USERNAME/alzheimer-mri-pipeline.git
cd alzheimer-mri-pipeline

2. Environment Setup

                                                                    bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

🔬 Core Highlights & Capabilities

    Sample-Efficient Learning: Exploits contrastive learning to extract high-utility latent representations even with limited clinical annotations.
    Modular Pipeline Design: Highly decoupled components allowing easy substitution of backbones, augmentations, and loss functions.
    Production-Ready Python Code: Modular, documented, and independent from fragile external dependencies.

📬 Contact & Inquiries

For technical inquiries, collaboration, or consulting on Computer Vision & Machine Learning pipelines:

    GitHub: @neftekhari-ai
    Email: neftekhari.ai@proton.me
