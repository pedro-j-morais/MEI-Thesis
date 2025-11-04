# 🧪 Experiments Overview

This directory contains all experimental implementations and evaluations conducted for the thesis on **Morph Attack Detection (MAD)**. Each subfolder represents a distinct experiment based on published research papers, including dataset preparation, model implementation, evaluation, and analysis.

---

## 📁 Experiment Folders

### 1. **1-SPL-MAD**
**Self-Paced Learning MAD**

Based on *Fang et al., 2022* (DOI: [10.1109/IJCB54206.2022.10008003](https://doi.org/10.1109/IJCB54206.2022.10008003))

This experiment implements an **unsupervised anomaly detection** approach for MAD using a self-paced learning mechanism. The model gradually includes more challenging samples during training to improve generalization in distinguishing bona fide facial representations from morphed images.

**Key Components:**
- Dataset preparation for FRLL-Morphs and SMDD
- Self-paced learning pipeline setup
- Equal Error Rate (EER) analysis across multiple datasets and morphing methods (AMSL, FaceMorpher, OpenCV, StyleGAN, WebMorph)

---

### 2. **2-MorDeephy-Benchmark**
**MorDeephy Framework Evaluation**

Based on *Medvedev et al., 2023* (DOI: [10.5220/0011606100003411](https://doi.org/10.5220/0011606100003411))

This experiment explores the **MorDeephy** benchmark framework, a publicly available protocol designed to evaluate MAD methods. Rather than providing a complete model implementation, it offers a structured pipeline where researchers can integrate and test their own models.

**Key Components:**
- Dataset download and extraction pipeline
- Image alignment and preprocessing
- Prediction simulation and benchmark evaluation
- Performance metrics calculation (APCER, BPCER, AUC)

---

### 3. **3-Attack-Agnostic-Features**
**Attack-Agnostic Feature Extraction for MAD**

Based on *Colbois et al., 2024* (DOI: [10.1109/IJCB62174.2024.10744532](https://doi.org/10.1109/IJCB62174.2024.10744532))

This experiment attempts to replicate a workflow investigating MAD using attack-agnostic features across FFHQ-Morphs, FRGC-Morphs, and FRLL-Morphs datasets. The goal was to reproduce the experimental pipeline, though the replication faced challenges due to code malfunctions and missing dataset components.

**Key Components:**
- Environment setup and dataset preparation
- Feature extraction pipeline configuration
- Dependency management and compatibility adjustments
- Documentation of replication challenges and limitations

**Status:** Partial implementation - full pipeline non-reproducible due to unresolved dependencies and missing datasets.

---

### 4. **5-ViT_SVM-OcclusionMapping**
**Vision Transformer + SVM with Occlusion Analysis**

Based on *Zhang et al., 2024* (DOI: [10.1109/CVPRW63382.2024.00158](https://doi.org/10.1109/CVPRW63382.2024.00158))

This experiment analyzes occlusion effects on facial MAD systems using a **Vision Transformer (ViT)** for feature extraction and a **Support Vector Machine (SVM)** for classification. The study investigates which facial regions are most critical for accurate detection.

**Key Components:**
- ViT-based feature extraction from facial images
- SVM classification for bona fide vs. morphed detection
- Occlusion mapping generation (grid-based and landmark-based)
- Sensitivity analysis to identify vulnerable facial regions
- Performance evaluation under controlled occlusion scenarios

---

## 🎯 Research Objectives

These experiments collectively aim to:
1. Evaluate state-of-the-art MAD approaches across diverse datasets
2. Understand the robustness of detection methods under various morphing techniques
3. Analyze the impact of occlusions on detection performance
4. Benchmark different architectural approaches (unsupervised learning, deep features, transformers)
5. Identify critical facial regions for morphing detection

---

## 📊 Datasets Used

- **FRLL-Morphs**: Face Research London Lab morphs dataset
- **SMDD**: Synthetic Face Morphing Attack Detection Development dataset
- **FFHQ-Morphs**: High-quality facial morphs based on FFHQ
- **FRGC-Morphs**: Face Recognition Grand Challenge morphs dataset
- Various morphing methods: AMSL, FaceMorpher, OpenCV, StyleGAN, WebMorph

---

## 🔧 Execution Environment

All experiments are designed for execution in **Google Colab** environments with GPU support. Each experiment folder contains detailed README files with specific setup instructions, workflow steps, and additional notes.
