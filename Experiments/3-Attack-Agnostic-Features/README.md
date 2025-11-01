# Colbois et al. (2024) – Morphing Attack Detection Experiment

This notebook attempts to replicate the workflow proposed by **Colbois et al. (2024)** (DOI: [10.1109/IJCB62174.2024.10744532](https://doi.org/10.1109/IJCB62174.2024.10744532)), which investigates Morphing Attack Detection (MAD) using the **FFHQ-Morphs**, **FRGC-Morphs** **FRLL-Morphs** datasets. The goal was to reproduce the experimental pipeline described in the publication, adapting the available resources to the local execution environment.

Despite extensive setup efforts, the replication was **not successfully completed** due to unresolved code malfunctions, dependency issues, and the unavailability of the regenerated **FRLL-Morphs** dataset required for full evaluation.

---

### Notebook Cell Explanations

1. **Initialize Environment**
   * Sets up the shell environment for running installation and system commands in Colab.

2. **Connect Google Drive**
   * Mounts Google Drive to access the shared **FFHQ dataset** directory required for data preparation.

3. **Download FFHQ-Morphs Dataset**
   * Attempts to download and extract the **FFHQ-Morphs** dataset into `/content/datasets/ffhq-morphs/FFHQ-Morphs/`.
   * This dataset was successfully retrieved, but later steps depended on unavailable components of the **FRLL-Morphs** dataset.

4. **Update Dataset Metadata**
   * Edits `dataset.csv` and `gathered_sets.csv` to align dataset paths with the local environment.
   * These adjustments ensure partial dataset compatibility but did not resolve downstream processing issues.

5. **Adjust Experiment Configuration**
   * Modifies experiment scripts to remove references to unsupported datasets and fix broken repository links in files like `external.smk`.

6. **Remove Inactive or Broken Extractors**
   * Disables nonfunctional extractors that caused runtime errors during data processing and feature extraction.

7. **Install Conda Environment (via CondaColab)**
   * Uses `condacolab` to replicate the dependency environment defined in the Colbois et al. implementation.
   * Several dependency mismatches and outdated package versions led to persistent installation failures.

8. **Pipeline Execution Attempt**
   * Attempts to execute the main experimental pipeline after configuration.
   * Execution was halted due to unresolved code malfunctions and missing dataset components.

---

### Outcome

Due to a combination of **unresolved code malfunctions**, **dependency errors**, and the **unavailability of the regenerated FRLL-Morphs dataset**, the replication attempt could not be completed.  
While the notebook successfully documents the setup and partial configuration process, the **full MAD evaluation** described by Colbois et al. (2024) remains **non-reproducible** under the current conditions.


Colbois, A., et al. (2024). *[Title of the original publication, if available]*.  
(Provide DOI or arXiv link once confirmed)
