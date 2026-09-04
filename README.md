# 🔬 Microscopy Nanowell Processor & AI Quantification Suite

A high-performance standalone GUI desktop tool designed to seamlessly process, align, multi-channel crop, and run deep-learning-based single-cell segmentation on large-scale stitched microscopy images of circular nanowell arrays.

---

## 🌟 Key Features

* **Automated Grid Alignment & Cropping**: Mathematical modeling and OpenCV computer vision algorithms calibrate central origin squares, rotation angles, and inter-well pitch.
* **Multi-Channel Synchronized Cropping**: Slices high-resolution Brightfield and fluorescence stitched slides into coordinate-indexed micro-wells across timepoints (`DayX`).
* **Deep Learning Cell Segmentation**: Powered by **Cellpose (`cpsam_v2`)** with GPU-accelerated batch inference (`CUDA 12.x`).
* **Multi-Mode Tracking**:
  * **Mode 1**: Single-cell isolation filter (`Count == 1` only).
  * **Mode 2**: Targeted coordinate tracking based on pre-existing Excel records.
  * **Mode 3**: High-throughput whole-array quantification.
* **Longitudinal Excel Database**: Automatically merges cross-day statistics into a standardized master file (`<WellName>_CellCount.xlsx`) with automated zero-collision backups.
* **Visual Validation**: Optional export of color-coded instance segmentation masks mapped directly to individual nanowell coordinates.
* **Safe Purge & Rollback Protection**: Prevents stale orphan crops when updating grid calibration layouts.

---

## 📁 Input Data Requirements (Raw Images)
Before running the processing engine, ensure your files are inside the same directory and adhere to the expected naming conventions. The software uses these patterns to automatically detect experimental groups.

* **Supported Formats:** `.tif` (Optimized for high-resolution stitched slides).
* **Supported Image names:** `Well[A-Z][0-9][0-9]_XXX_[Channel].tif` ( XXX can be anything. [Channel] has to be one of them in below)

| Channel | Standardized Channel Code | Optical Channel & Configuration Description |
| :--- | :--- | :--- |
| `RGB` | `RGB` | **Merged / Overlay Status**: A composite color image representing all captured channels merged together into a single frame for global previewing. |
| <code>20&nbsp;Phase</code>| `BF` | **Brightfield Channel**: Transmitted light imaging mode using a **20X** objective lens, primarily utilized for structural tracking and grid calibration. Name needs to end with '_20X Phase' to be renamed as BF. |
| `GFP` | `GFP` | **eGFP Fluorescence Channel**: Green fluorescent protein detection channel captured at a **20X** objective magnification, used to observe target cellular expressions. |
| `mCherry` | `mCherry` | **mCherry Fluorescence Channel**: Red fluorescent protein detection channel captured at a **20X** objective magnification, optimized for tracking red fluorophore expressions. |
> ⚠️ **Prerequisite Rule:** To perform any cropping operations, the **BF** (Brightfield) channel image **must be present** in the directory as it is strictly required for mathematical grid calibration. All other fluorescence channels (`GFP`, `mCherry`, `RGB`) are completely optional and can be included or omitted based on your experimental dataset.


## 🚀 Workflow & Usage

### 🎛️ Page 1: Filename Standardization & Nanowell Cropping

1. **Step 1: Standardize Raw Names**  
   Select the raw directory and click **`Standardize File Names`**. Files will be uniformly renamed to:  
   `[WellName]_Day[Index]_[Channel].tif` *(e.g., `A02_Day1_BF.tif`)*.
2. **Step 2: Calibrate Grid Alignment**  
   Click **`Calculate Initial Parameters`** to auto-detect the square origin marker and grid rotation angle, or fine-tune parameters manually. Click **`Visualize / Update Preview`** to inspect alignment overlays on the interactive canvas.
3. **Step 3: Crop & Export**  
   Select output resolution (Default: `380 px`) and click **`✂️ Crop & Export All Nanowells`**.
   *The program automatically applies the BF grid alignment across all detected fluorescence channels to batch-crop matching positions in a single run.*
   *If cropped files already exist for the selected Well and Day, the tool prompts you to safely purge old crops before writing new coordinates.*

---

### 🧠 Page 2: AI Cell Segmentation & Quantification

Click **`Next: Image Analysis ➡️`** at the bottom to switch to the AI workspace.

1. **Parameter Inheritance**: `Processed Wells Dir`, `Well Name`, and `Day Index` automatically sync from Page 1.
2. **Select Analysis Mode**:
   * **Mode 1**: Process and record **single-cell nanowells only** (`Count == 1`).
   * **Mode 2**: Process **only designated coordinates** listed in the existing `<WellName>_CellCount.xlsx`. *(Auto-selected as default if an existing database is found)*.
   * **Mode 3**: Process **all cropped nanowells** within the selected Well directory.
3. **Cell Diameter (px)**:
   * **`40 px (Default)`**: Recommended baseline for Cellpose `cpsam_v2` inference.
   * **`Other`**: Enter a custom diameter *(e.g., `25`, `50`)* for specific cell lines.
4. **Validation Masks (Optional)**:
   * Check **`Save validation mask images`** to export paired raw inputs and colored instance masks.
5. **Execute Pipeline**:
   * Click **`🧠 Run AI Segmentation & Update Excel`**. The background thread handles GPU batch inference (`batch_size=16`) with real-time progress updates.

---

## 📤 Output Directory Structure

All outputs are structured under a single parent directory named **`Processed Wells`**:

```text
Processed Wells/
├── A02_CellCount.xlsx                  <-- Longitudinal Master Database
├── AI Segmentations/                   <-- (Optional) Validation Visualizations
│   └── A02_Day1_Masks/
│       ├── A02_R0_C0_Day1_BF.png        <-- Raw Input
│       ├── A02_R0_C0_Day1_BF_mask.png   <-- Multi-color Instance Mask
│       ├── A02_R0_C1_Day1_BF.png
│       └── A02_R0_C1_Day1_BF_mask.png
│
└── A02/                                <-- Cropped Channel Repositories
    ├── BF/
    │   ├── A02_R0_C0_Day1_BF.png       <-- (R0_C0: Origin Square)
    │   ├── A02_R0_C1_Day1_BF.png
    │   ├── A02_R0_C1_Day2_BF.png
    │   └── A02_R5_C10_Day1_BF.png
    ├── GFP/
    │   ├── A02_R0_C0_Day1_GFP.png
    │   └── A02_R0_C1_Day1_GFP.png
    └── mCherry/
        ├── A02_R0_C0_Day1_mCherry.png
        └── A02_R0_C1_Day1_mCherry.png

* **R** stands for **Row Index** (vertical position starting from 0)
* **C** stands for **Column Index** (horizontal position starting from 0)
(0, 0) => R0_C0 is the Central Orientation Rectangle (Origin)
(0, 1) => R0_C1
(0, -1)=> R0_C_1

```

## 📊 Excel Quantification Output (_CellCount.xlsx)
The system aggregates longitudinal metrics into a single master spreadsheet per well (e.g., A02_CellCount.xlsx):

| Coordinate | Day1_Cell_Count | Day1_Cell_Area | Day2_Cell_Count | Day2_Cell_Area | Day3_Cell_Count | ... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R0_C1` | 1 | 480 | 2 | 1208 | 4 | ... |
| `R0_C2` | 0 | 0 | 0 | 0 | 0 | ... |
| `R0_C3` | 1 | 503 | 1 | 510 | 2 | ... |

> Zero-Collision Backup Engine: If an open Excel file or I/O lock interrupts the merge process, newly calculated metrics are saved to an isolated timestamped backup:
`Processed Wells/<WellName>_CellCount_Day<Day>_backup_<YYYYMMDD_HHMMSS>.xlsx`

<img width="1584" height="1049" alt="page1" src="https://github.com/user-attachments/assets/d1561877-1c6b-4bf0-9196-f5eb8ae68529" />
<img width="1594" height="1054" alt="page2" src="https://github.com/user-attachments/assets/4be7f5d2-437c-43a3-af0d-d66c62e58686" />


Recommended tif exporting using NIS:
<img width="1277" height="857" alt="NIS_export" src="https://github.com/user-attachments/assets/a0c885a1-2759-43a8-99a7-f838d23832c4" />


