# 🧠 Colbois2024 Attack-Agnostic Features – Project Overview

This experiment attempts to replicate the workflow proposed by *Colbois et al., 2024* (DOI: [10.1109/IJCB62174.2024.10744532](https://doi.org/10.1109/IJCB62174.2024.10744532)), which investigates **Morphing Attack Detection (MAD)** using attack-agnostic features across the **FFHQ-Morphs**, **FRGC-Morphs**, and **FRLL-Morphs** datasets. The goal was to reproduce the experimental pipeline described in the publication, adapting the available resources to the local execution environment.

Despite extensive setup efforts, the replication was **not successfully completed** due to unresolved code malfunctions, dependency issues, and the unavailability of the regenerated **FRLL-Morphs** dataset required for full evaluation.

---

## 📁 Notebook Summaries

### 1. **`Colbois2024.ipynb`**
#### 🔍 Objective:
Attempts to replicate the attack-agnostic feature extraction and MAD evaluation pipeline proposed by Colbois et al. (2024), including dataset preparation, feature extraction, and model evaluation.

#### 🧬 Code Description:
- **Initialize Environment**:
  - Sets up the shell environment for running installation and system commands in Colab.
- **Connect Google Drive**:
  - Mounts Google Drive to access the shared **FFHQ dataset** directory required for data preparation.
- **Download FFHQ-Morphs Dataset**:
  - Attempts to download and extract the **FFHQ-Morphs** dataset into `/content/datasets/ffhq-morphs/FFHQ-Morphs/`.
  - This dataset was successfully retrieved, but later steps depended on unavailable components of the **FRLL-Morphs** dataset.
- **Update Dataset Metadata**:
  - Edits `dataset.csv` and `gathered_sets.csv` to align dataset paths with the local environment.
  - These adjustments ensure partial dataset compatibility but did not resolve downstream processing issues.
- **Adjust Experiment Configuration**:
  - Modifies experiment scripts to remove references to unsupported datasets and fix broken repository links in files like `external.smk`.
- **Remove Inactive or Broken Extractors**:
  - Disables nonfunctional extractors that caused runtime errors during data processing and feature extraction.
- **Install Conda Environment (via CondaColab)**:
  - Uses `condacolab` to replicate the dependency environment defined in the Colbois et al. implementation.
  - Several dependency mismatches and outdated package versions led to persistent installation failures.
- **Pipeline Execution Attempt**:
  - Attempts to execute the main experimental pipeline after configuration.
  - Execution was halted due to unresolved code malfunctions and missing dataset components.

---

## 🧪 Suggested Workflow

1. **Environment Setup**:
   - Initialize the Colab environment and mount Google Drive for dataset access.
2. **Dataset Preparation**:
   - Download and extract the FFHQ-Morphs dataset and configure metadata files.
3. **Configuration Adjustments**:
   - Modify experiment scripts to align with available datasets and fix broken dependencies.
4. **Install Dependencies**:
   - Set up the Conda environment using CondaColab with required packages.
5. **Execute Pipeline**:
   - Run the feature extraction and MAD evaluation pipeline (currently non-functional).

---

## 🧩 Additional Notes
- **Replication Status**: The full MAD evaluation described by Colbois et al. (2024) remains **non-reproducible** under current conditions.
- **Known Issues**:
  - Unresolved code malfunctions in the original implementation.
  - Dependency errors and outdated package versions.
  - Unavailability of the regenerated **FRLL-Morphs** dataset required for complete evaluation.
- While the notebook successfully documents the setup and partial configuration process, the pipeline cannot be executed end-to-end.
- Designed for execution in **Google Colab** environments.
