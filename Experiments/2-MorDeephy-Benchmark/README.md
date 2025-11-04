# 🧠 MorDeephy Benchmark – Project Overview

This experiment explores the **MorDeephy** framework from *Medvedev et al., 2023* (DOI: [10.5220/0011606100003411](https://doi.org/10.5220/0011606100003411)), a publicly available benchmark designed to evaluate **Morphing Attack Detection (MAD)** methods. The repository does not offer a complete, ready-to-use implementation of the model described in the publication; instead, it provides a structured protocol where researchers can integrate and test their own models. A custom Jupyter notebook was developed to verify this demonstration pipeline, following key steps from data acquisition and alignment to the final calculation of performance metrics based on the prediction scores obtained.

---

## 📁 Notebook Summaries

### 1. **`MorDeephy.ipynb`**
#### 🔍 Objective:
Demonstrates the MorDeephy benchmark pipeline, including dataset preparation, image alignment, prediction simulation, and performance evaluation for Morphing Attack Detection systems.

#### 🧬 Code Description:
- **Clone Repository**:
  - Clones the `MorDeephy` project from its GitHub repository to the local environment.
- **Download Datasets**:
  - Executes the `download_data.py` script to fetch the required facial image datasets for the benchmark.
  - The script successfully downloads several datasets, but the log indicates a failure for the AR Face Database.
- **Extract Data**:
  - Installs the `unlzw` and `patool` packages, which are required for file decompression.
  - Runs the `data_extract.py` script to extract the downloaded archives into the `data_extracted/` directory.
- **Align Images**:
  - Runs the `align_protocol_insf.py` script to preprocess and align the faces in the images according to the benchmark's protocol.
  - The script fails during execution with an `OpenCV error` when it attempts to process a file from a dataset that was not successfully extracted in the previous step.
- **Prediction Simulation**:
  - Runs the `sd_demo_extracting_predictions.py` script to simulate a model's output by generating illustrative prediction scores.
  - This serves to illustrate the data structure required by the benchmark, showing how prediction scores and ground-truth labels should be formatted for evaluation. It does not perform actual inference with a deep learning model.
- **Benchmark Execution**:
  - Executes `sd_benchmark_model.py`, which processes the simulated predictions generated in the prior step.
  - Computes and reports evaluation metrics, including APCER, BPCER, and an illustrative AUC ROC of 0.949. This output demonstrates the calculation functionality of the benchmark framework, showing the results format for when a real model's predictions are supplied.

---

## 🧪 Suggested Workflow

1. **Clone and Setup**:
   - Clone the MorDeephy repository and install required dependencies.
2. **Download and Extract Datasets**:
   - Download benchmark datasets and extract them to the appropriate directories.
3. **Align Images**:
   - Preprocess and align facial images according to the benchmark protocol.
4. **Generate Predictions**:
   - Run your MAD model or use the demo prediction script to generate scores.
5. **Evaluate Performance**:
   - Execute the benchmark evaluation script to compute APCER, BPCER, and AUC metrics.

---

## 🧩 Additional Notes
- The framework is designed for **benchmark integration**, not as a standalone MAD solution.
- Researchers must integrate their own models to generate actual predictions.
- Some datasets may fail to download or extract; verify data availability before running the full pipeline.
- The demonstration uses simulated predictions to illustrate the expected data format and evaluation process.
