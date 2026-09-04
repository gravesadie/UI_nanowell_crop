# ==============================================================================
# AI Analysis & Segmentation Engine (Cellpose cpsam_v2) - Batch Accelerated
# ==============================================================================
"""
if export masks:

Processed Wells/
├── AI Segmentations/
│   └── A02_Time1_Masks/
│       ├── A02_R0_C1_Time1_BF.png        <-- original input image
│       ├── A02_R0_C1_Time1_BF_mask.png   <-- Corresponding mask (multiple cells marked with different color)
│       ├── A02_R0_C2_Time1_BF.png
│       └── A02_R0_C2_Time1_BF_mask.png
└── A02_CellCount.xlsx

"""

import os
import re
import cv2
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts integer instance mask to an RGB visualization with distinct vibrant colors per cell.
    Background (0) remains pitch black.
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_cells = np.unique(mask[mask > 0])
    if len(unique_cells) == 0:
        return colored_mask

    # Generate distinct HSV colors and convert to BGR for OpenCV saving
    for cell_id in unique_cells:
        # Golden ratio hue distribution for maximum color contrast
        hue = int((cell_id * 0.618033988749895 % 1.0) * 179)
        hsv_pixel = np.uint8([[[hue, 220, 255]]])
        bgr_color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        
        colored_mask[mask == cell_id] = bgr_color

    return colored_mask


def execute_ai_segmentation(processed_wells_dir: str, well_name: str, time: str,
                            mode: int, model_dir: str, save_masks: bool = False, 
                            cell_diameter: float = 30.0, log_callback=print, 
                            progress_callback=None, batch_size: int = 16):
    """
    Executes Cellpose segmentation on cropped Brightfield nanowell images using Batch Inference.
    Modes:
      1: Single-cell filter mode (finds and exports wells containing exactly 1 cell).
      2: Process designated coordinates from an Excel file in 'Processed Wells'.
      3: Process all valid cropped BF images matching well_name and time.
    Outputs/Updates Excel: Processed Wells/<well_name>_CellCount.xlsx
    Optionally exports debug pairs to: Processed Wells/AI Segmentations/<well_name>_Time<time>_Masks/
    """
    if not CELLPOSE_AVAILABLE:
        log_callback("❌ [ERROR]: 'cellpose' package is not installed in the active environment.")
        return

    processed_wells_dir = processed_wells_dir.strip().replace('\\', '/')
    well_name = well_name.strip()
    time = time.strip()

    bf_folder = os.path.join(processed_wells_dir, well_name, "BF")
    if not os.path.exists(bf_folder):
        log_callback(f"❌ [ERROR]: Bright-field (BF) directory not found at: {bf_folder}")
        return

    # 1. Setup Mask Export Directory if requested
    mask_export_dir = ""
    if save_masks:
        mask_export_dir = os.path.join(processed_wells_dir, "AI Segmentations", f"{well_name}_Time{time}_Masks")
        Path(mask_export_dir).mkdir(parents=True, exist_ok=True)
        log_callback(f"📁 [MASKS]: Exporting validation masks to:\n{mask_export_dir}")

    # 2. Locate and load the cpsam_v2 model
    model_path = os.path.join(model_dir, "cpsam_v2")
    if not os.path.exists(model_path):
        log_callback(f"❌ [ERROR]: Pre-trained model 'cpsam_v2' not found under: {model_dir}")
        return

    log_callback(f"[AI MODEL]: Initializing Cellpose model from '{model_path}'...")
    try:
        model = models.CellposeModel(pretrained_model=model_path, gpu=True)
        log_callback("✅ [AI MODEL]: Model loaded successfully (GPU acceleration active).")
    except Exception:
        model = models.CellposeModel(pretrained_model=model_path, gpu=False)
        log_callback("⚠️ [AI MODEL]: CUDA GPU unavailable. Model running in CPU mode.")

    # 3. Gather image candidates matching pattern: <well_name>_<coordinate>_Time<time>_BF.png
    pattern = rf"^{well_name}_(R[-]?\d+_C[-]?\d+)_Time{time}_BF\.png$"
    available_files = {}
    for fname in os.listdir(bf_folder):
        m = re.match(pattern, fname)
        if m:
            coord = m.group(1)
            available_files[coord] = os.path.join(bf_folder, fname)

    if not available_files:
        log_callback(f"⚠️ [WARNING]: No Bright-field images found for Well '{well_name}' Time '{time}'.")
        return

    # 4. Filter target coordinates based on chosen mode
    target_coords = list(available_files.keys())
    excel_cell_count = os.path.join(processed_wells_dir, f"{well_name}_CellCount.xlsx")

    if mode == 2:
        if not os.path.exists(excel_cell_count):
            log_callback(f"❌ [ERROR]: No coordinate Excel file found at: {excel_cell_count}")
            return
        coord_excel_path = excel_cell_count

        log_callback(f"[INPUT]: Reading coordinate list from: {coord_excel_path}")
        try:
            df_in = pd.read_excel(coord_excel_path)
            coord_col = None
            for c in df_in.columns:
                if str(c).strip().lower() in ["coordinate", "coordinates", "address"]:
                    coord_col = c
                    break
            if coord_col is None:
                log_callback(f"❌ [ERROR]: No coordinate information found in the Excel file {coord_excel_path}.")
                return

            filtered_coords = df_in[coord_col].dropna().astype(str).tolist()
            target_coords = [c for c in filtered_coords if c in available_files]
            log_callback(f"ℹ️ [INFO]: Filtered {len(target_coords)} coordinates from Excel definition.")
        except Exception as e:
            log_callback(f"❌ [ERROR]: Failed parsing Excel coordinates: {e}")
            return

    # 5. Batch Inference Pipeline
    total_nodes = len(target_coords)
    log_callback(f"[RUNNING]: Segmenting {total_nodes} nanowell images with Batch Size = {batch_size}...")
    results = []
    processed_count = 0

    for i in range(0, total_nodes, batch_size):
        batch_coords = target_coords[i:i + batch_size]
        batch_imgs = []
        raw_imgs = []
        valid_batch_coords = []
        raw_paths = []

        for coord in batch_coords:
            img_path = available_files[coord]
            raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if raw_img is None:
                continue

            if len(raw_img.shape) == 3:
                img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY if raw_img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
            else:
                img_gray = raw_img

            batch_imgs.append(img_gray)
            raw_imgs.append(raw_img)
            valid_batch_coords.append(coord)
            raw_paths.append(img_path)

        if not batch_imgs:
            continue

        masks_batch, _, _ = model.eval(
            batch_imgs,
            batch_size=len(batch_imgs),
            diameter=cell_diameter,
            channels=[0, 0],
            resample=False,
            normalize=True
        )

        if isinstance(masks_batch, np.ndarray) and masks_batch.ndim == 3:
            masks_list = [masks_batch[j] for j in range(masks_batch.shape[0])]
        elif isinstance(masks_batch, list):
            masks_list = masks_batch
        else:
            masks_list = [masks_batch]

        for coord, masks, raw_img, src_path in zip(valid_batch_coords, masks_list, raw_imgs, raw_paths):
            processed_count += 1

            cell_count = int(len(np.unique(masks[masks > 0]))) if masks is not None else 0
            total_cell_area = int(np.count_nonzero(masks)) if masks is not None else 0

            # Mode 1: Skip non-single-cell wells
            if mode == 1 and cell_count != 1:
                continue

            results.append({
                "Coordinate": coord,
                f"Time{time}_Cell_Count": cell_count,
                f"Time{time}_Cell_Area": total_cell_area
            })

            # Export validation image pair if checkbox was checked
            if save_masks and mask_export_dir:
                base_name = os.path.splitext(os.path.basename(src_path))[0]
                
                # 1. Save original AI Input
                out_raw_path = os.path.join(mask_export_dir, f"{base_name}.png")
                cv2.imwrite(out_raw_path, raw_img)
                
                # 2. Save colored Mask (<name>_mask.png)
                colored_mask = colorize_mask(masks if masks is not None else np.zeros(raw_img.shape[:2], dtype=np.uint16))
                out_mask_path = os.path.join(mask_export_dir, f"{base_name}_mask.png")
                cv2.imwrite(out_mask_path, colored_mask)

        if progress_callback:
            progress_callback(processed_count, total_nodes)

        log_callback(f" -> Batch Processed [{processed_count}/{total_nodes}] ({processed_count / total_nodes * 100:.1f}%)")

    if not results:
        log_callback("⚠️ [WARNING]: No wells met the analysis criteria. Excel file not modified.")
        return

    # 6. Merge and Write to Well-specific Excel
    output_excel_path = excel_cell_count
    df_new = pd.DataFrame(results)

    if os.path.exists(output_excel_path):
        log_callback(f"[EXCEL]: Existing database found at '{output_excel_path}'. Merging columns...")
        try:
            df_existing = pd.read_excel(output_excel_path)
            df_merged = pd.merge(df_existing, df_new, on="Coordinate", how="outer")
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
            df_merged.to_excel(output_excel_path, index=False)
            log_callback(f"✅ [SUCCESS]: Updated existing workbook successfully with Time {time} metrics.")
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_excel_name = f"{well_name}_Time{time}_backup_{timestamp}.xlsx"
            backup_excel_path = os.path.join(processed_wells_dir, backup_excel_name)

            log_callback(f"⚠️ [WARNING]: Merge failed ({e}). Saving to separate backup file instead.")
            df_new.to_excel(backup_excel_path, index=False)
            log_callback(f"📁 [BACKUP CREATED]: Saved new data to:\n{backup_excel_path}")
    else:
        df_new.to_excel(output_excel_path, index=False)
        log_callback(f"✅ [SUCCESS]: Generated new analysis workbook at: {output_excel_path}")

    log_callback(f"🏁 [COMPLETE]: AI Analysis Task Finished. Processed {len(results)} valid nodes.")