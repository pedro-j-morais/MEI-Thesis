
# Dataset and Heatmap Builder Scripts

This document describes how the three core scripts work and how they fit together in the workflow for preparing datasets, generating occlusion variants, and visualizing occlusion heatmaps.

## 1. crop_dataset_builder

### Purpose

`crop_dataset_builder` prepares a clean, standardized dataset by
cropping and optionally resizing facial images before any occlusion
experiments.\
It ensures that every image in the dataset has a consistent face region
and size.

### Key Features

- **Face detection & cropping**: Uses MTCNN to detect the face and crop the image to the bounding box.
- **Preserves folder structure**: Cropped faces are saved in the same relative directory tree under the output folder.
- **Multiprocessing**: Optional parallel processing for large datasets.
- **Detailed logging**: Records progress, warnings, and processing times to both console and log file.

### Requirements

- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/)
- [MTCNN](https://pypi.org/project/mtcnn/)
- [Pillow](https://pillow.readthedocs.io/)
- NumPy

Install dependencies:

```bash
pip install tensorflow mtcnn pillow numpy
```

### Arguments

| Argument            | Type    | Required | Default                 | Description                                                                                     |
|---------------------|---------|---------|-------------------------|-------------------------------------------------------------------------------------------------|
| `--input_path`      | Path    | Yes     | —                       | Root folder containing images to process.                                                       |
| `--output_path`     | Path    | Yes     | —                       | Root folder where cropped faces will be saved (folder structure is preserved).                  |
| `--multiprocessing` | Flag    | No      | Disabled                | Enable multiprocessing for faster processing.                                                   |
| `--workers`         | Integer | No      | Number of CPU cores     | Number of worker processes when multiprocessing is enabled.                                     |


### Example Command

``` bash
python crop_dataset_builder.py \
    --input /path/to/raw_dataset \
    --output /path/to/cropped_dataset \
    --multiprocessing --workers 12
```
------------------------------------------------------------------------

## 2. occlusion_dataset_builder

### Purpose

`occlusion_dataset_builder` generates **occluded variants** of each
cropped image to simulate different types of facial occlusion.\
These occlusions are used to test the robustness of morph attack
detection (MAD) models.

### Types of Occlusions

-   **Grid-Based Occlusion**: Divides the face into an `R×C` grid and
    covers one cell (e.g., black rectangles) at a time to hide parts of
    the face.
-   **Landmark-Based Occlusion**: Uses facial landmarks (eyes, nose,
    mouth, etc.) and covers specific facial regions.

### Key Features

- **Grid occlusion**: Divides each image into a grid and occludes one cell at a time.  
- **Landmark occlusion**: Uses Dlib 68-point facial landmarks to occlude semantic facial regions.  
- **Both modes**: Optionally generates grid and landmark occlusions in a single run.  
- **Preserves folder structure**: Output images maintain the relative folder paths of the original dataset.  
- **Multiprocessing support**: Speeds up dataset generation on large datasets.  
- **Logging**: Tracks processing progress, warnings, and errors in a timestamped log file.  
- **Automatic Dlib model download**: Downloads and extracts the 68-point shape predictor if missing.

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)

Install dependencies:

```bash
pip install numpy opencv-python dlib
```

### Arguments

| Argument                  | Type / Choices                                | Required | Default                 | Description                                                                                                         |
|---------------------------|-----------------------------------------------|---------|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| `--mode`                  | Choice: `grid` \| `landmark` \| `both`         | Yes     | —                       | Which occlusion(s) to generate:<br>• `grid` – grid-based occlusion only<br>• `landmark` – landmark-based occlusion only<br>• `both` – run both methods. |
| `--input_path`            | Path                                          | Yes     | —                       | Root folder containing input images (nested subfolders supported).                                                  |
| `--output_path`           | Path                                          | Yes     | —                       | Root folder where occluded images will be saved. Subfolders `grid-occlusion` and/or `landmark-occlusion` are created automatically. |
| `--grid_rows_count`       | Integer                                       | No      | 4                       | Number of rows to divide the image grid for grid occlusion.                                                         |
| `--grid_columns_count`    | Integer                                       | No      | 4                       | Number of columns to divide the image grid for grid occlusion.                                                      |
| `--landmark_precision`    | Choice: `pixel` \| `bbox`                      | No      | `pixel`                 | Landmark occlusion method:<br>• `pixel` – mask the exact polygon of each region<br>• `bbox` – mask the bounding box of each region. |
| `--landmark_scale_factor` | Float                                         | No      | 1.5                     | Scale factor to expand or shrink the landmark polygons or bounding boxes.                                           |
| `--multiprocessing`       | Flag                                         | No      | Disabled                | Enable multiprocessing for faster dataset creation.                                                                 |
| `--workers`               | Integer                                      | No      | Number of CPU cores     | Number of worker processes to use when `--multiprocessing` is enabled.                                              |

### Example Command

``` bash
python occlusion_dataset_builder.py \
		--mode both \
		--input /path/to/cropped_dataset \
		--output /path/to/occluded_dataset \
		--grid_rows_count 4 \
		--grid_columns_count 4 \
		--landmark_precision bbox \
		--landmark_scale_factor 1.5 \
		--multiprocessing --workers 12
```

------------------------------------------------------------------------

## 3. occlusion_heatmaps_builder

### Purpose

`occlusion_heatmaps_builder` creates **visual heatmaps** that highlight
which parts of a face influence the model's prediction.\
It combines the prediction scores from regular, grid-occluded, and
landmark-occluded images.

### Key Features

- **Flexible score parsing**: Supports “regular”, “grid”, and “landmark” score files in CSV-like format.  
- **Grid-based heatmaps**: Builds per-image heatmaps from grid occlusion scores.  
- **Landmark-based heatmaps**: Builds per-image heatmaps using facial landmark occlusion scores.  
- **Composite heatmaps**: Optionally combine grid and landmark heatmaps.  
- **Automatic Dlib landmark detection**: Maps landmark regions to image coordinates.  
- **Heatmap modes**: Raw occlusion scores or absolute difference from baseline (“diff”).  
- **Visualization**: Overlays heatmaps on original images with grid lines, bounding boxes, and cell values.  
- **Multiprocessing support**: Efficient processing of large datasets.  
- **Logging and indexing**: Tracks progress, warnings, and errors in a timestamped log file.  
- **Automatic Dlib model download**: Downloads the 68-point shape predictor if missing.


### How It Works

1.  **Input Score Files**
    -   **Regular Scores**: Model predictions on the original
        (non-occluded) images.
    -   **Grid Scores**: Predictions for each grid-cell occluded
        variant.
    -   **Landmark Scores**: Predictions for each landmark-occluded
        variant.
2.  **Parsing & Matching**
    -   Matches each occluded image back to its corresponding original image using flexible path matching and regex patterns.
3.  **Heatmap Generation**
    -   Builds a per-image grid of scores.
    -   Supports two visualization modes:
        -   **occluded**: Shows raw model prediction when a region is
            occluded.
        -   **diff**: Shows absolute difference from the regular
            (baseline) score.
4.  **Overlay**
    -   Overlays the heatmap on top of the original image.
    -   Draws landmark bounding boxes.
    -   Supports combination of grid and landmark heatmaps.
5.  **Optional Landmark Detection**
    -   Uses dlib's 68-point shape predictor to automatically locate facial regions if more accurate landmark mapping is desired.

### Output

-   PNG heatmaps for:
    -   Grid occlusion
    -   Landmark occlusion
    -   Composite Heatmaps (grid + landmark)

### Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Dlib](http://dlib.net/)
- [tqdm](https://tqdm.github.io/)

Install dependencies:

```bash
pip install numpy opencv-python matplotlib dlib tqdm
```

### Arguments
| Argument                  | Type / Choices                                | Required | Default         | Description                                                                                                         |
|---------------------------|-----------------------------------------------|---------|----------------|---------------------------------------------------------------------------------------------------------------------|
| `--regular`               | Path                                          | Yes     | —              | Path to regular (baseline) score file.                                                                              |
| `--grid`                  | Path                                          | No      | —              | Path to grid occlusion score file.                                                                                  |
| `--landmark`              | Path                                          | No      | —              | Path to landmark occlusion score file.                                                                              |
| `--output`                | Path                                          | Yes     | —              | Output folder where overlay images will be saved.                                                                  |
| `--heatmap-mode`          | Choice: `occluded` \| `diff`                 | No      | `occluded`     | How to fill each cell:<br>• `occluded`: show raw occlusion score<br>• `diff`: show absolute difference from baseline |
| `--alpha`                 | Float                                         | No      | 0.6            | Transparency of the heatmap overlay.                                                                               |
| `--colormap`              | String                                        | No      | `hot`          | Matplotlib colormap name.                                                                                           |
| `--multiprocessing`       | Flag                                          | No      | Disabled       | Enable multiprocessing for faster processing.                                                                      |
| `--workers`               | Integer                                       | No      | 8              | Number of worker processes when multiprocessing is enabled.                                                        |
| `--log-level`             | String                                        | No      | `INFO`         | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                                                              |


### Example Command

``` bash
python occlusion_heatmaps_builder.py \
    --regular /path/to/regular_scores.csv \
    --grid /path/to/grid_scores.csv \
    --landmark /path/to/landmark_scores.csv \
    --output /path/to/output_heatmaps \
    --heatmap-mode diff \
    --multiprocessing --workers 12
```
------------------------------------------------------------------------

## End-to-End Workflow

1.  **Crop**: Run `crop_dataset_builder` to obtain clean, standardized face crops.
2.  **Occlude**: Use `occlusion_dataset_builder` to generate grid- and landmark-occluded variants.
3.  **Predict**: Pass all images (regular + occluded) through your MAD
    model to produce prediction score files.
4.  **Visualize**: Run `occlusion_heatmaps_builder` with the
    regular/grid/landmark score files to create visual heatmaps.

This pipeline helps evaluate which facial regions most influence the
model's ability to detect morphing attacks and ensures a reproducible,
automated analysis.



## Important Notes

-   **Performance**: Can use multiprocessing (`--multiprocessing --workers 8`) for large datasets.
-   **Storage**: Avoid writing directly to Google Drive during processing. Save locally, zip, then upload if needed.
