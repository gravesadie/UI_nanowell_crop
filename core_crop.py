import os
import re
import math
import cv2
import numpy as np
from pathlib import Path

# Standard microscope channel mappings
CHANNEL_MAP = {
    '20X Phase': 'BF',
    'mCherry': 'mCherry',
    'GFP': 'GFP',
    'RGB': 'RGB'
}

def load_bf_image(img_dir: str, well_name: str, time: str):
    """
    Loads the Brightfield tile matrix image in 8-bit mode.
    Returns (cached_bgr, cached_gray, full_path, error_message).
    """
    img_dir = img_dir.strip().replace('\\', '/')
    well_name = well_name.strip()
    time = time.strip()

    bf_name = f"{well_name}_Time{time}_BF.tif"
    full_path = os.path.join(img_dir, bf_name)

    if not os.path.exists(full_path):
        return None, None, full_path, f"Target Brightfield image not found at:\n{full_path}"

    try:
        cached_bgr = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if cached_bgr is None:
            return None, None, full_path, f"Failed to decode image file:\n{full_path}"

        if len(cached_bgr.shape) == 2:
            cached_gray = cached_bgr.copy()
            cached_bgr = cv2.cvtColor(cached_bgr, cv2.COLOR_GRAY2BGR)
        else:
            cached_gray = cv2.cvtColor(cached_bgr, cv2.COLOR_BGR2GRAY)

        return cached_bgr, cached_gray, full_path, None
    except Exception as e:
        return None, None, full_path, f"Critical read error: {e}"


def rename_raw_files(directory: str, time: str, log_callback=print):
    """
    Standardizes raw microscope export filenames in the target directory.
    """
    directory = directory.strip().replace('\\', '/')
    time = time.strip()
    match_count = 0

    log_callback(f"[START]: Scanning files inside directory: {directory}")
    try:
        files = os.listdir(directory)
        for filename in files:
            if not filename.lower().endswith('.tif'):
                continue

            match_check_good = re.match(r"[A-Z]\d{2}_Time\d+_(.*)\.tif$", filename)
            if match_check_good and match_check_good.group(1) in {"RGB", "BF", "GFP", "mCherry"}:
                match_count += 1
                continue

            match_check = re.match(r"^Well([A-Z]\d{2})_(.*)_(.*)\.tif$", filename)
            if match_check:
                well, rest = match_check.group(1), match_check.group(3)
                channel = CHANNEL_MAP.get(rest, "UNKNOWN")
                new_filename = f"{well}_Time{time}_{channel}.tif"
                match_count += 1
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                log_callback(f"-> Renamed successfully: '{filename}' to '{new_filename}'")

        if match_count > 0:
            log_callback(f"✅ [FINISH]: Rename Task Complete. {match_count} total file(s) structured.")
        else:
            log_callback("❌ [WARNING]: No matching files found. Example accepted name: WellA02_XXX_20X Phase.tif")
    except Exception as e:
        log_callback(f"❌ [ERROR]: Rename engine failed: {e}")


def detect_center_square(cached_gray: np.ndarray, sq_len: int, roi_size: int = 6000):
    """
    Identifies the central square origin within a localized ROI.
    Returns (rect_center, cached_binary_crop, error_msg).
    """
    img_h, img_w = cached_gray.shape[:2]
    img_cx, img_cy = img_w // 2, img_h // 2

    half_roi = roi_size // 2
    roi_x1 = max(0, img_cx - half_roi)
    roi_y1 = max(0, img_cy - half_roi)
    roi_x2 = min(img_w, img_cx + half_roi)
    roi_y2 = min(img_h, img_cy + half_roi)

    roi_gray = cached_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    binary_fill_wt = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_fil_bk = cv2.morphologyEx(binary_fill_wt, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_fil_bk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (sq_len - 60) ** 2
    max_area = (sq_len + 60) ** 2
    best_score = float('inf')
    rect_center = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        (cx, cy), (w, h), _ = cv2.minAreaRect(cnt)
        if h == 0 or w == 0:
            continue

        rect_box_area = w * h
        extent = float(area) / rect_box_area
        if extent < 0.82:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if not (4 <= len(approx) <= 8):
            continue

        aspect_ratio = float(w) / h
        score = abs(1.0 - aspect_ratio)
        if 0.85 <= aspect_ratio <= 1.15 and score < best_score:
            best_score = score
            rect_center = (int(cx + roi_x1), int(cy + roi_y1))

    return rect_center, binary_fil_bk, None if rect_center else "Square marker not found in ROI."


def detect_array_angle(cached_gray: np.ndarray, nanowell_r: int, roi_rotation: int = 12000):
    """
    Estimates dominant grid rotation angle using downsampled HoughCircles and RANSAC pairs.
    """
    img_h, img_w = cached_gray.shape[:2]
    img_cx, img_cy = img_w // 2, img_h // 2

    half_roi = roi_rotation // 2
    roi_r_x1 = max(0, img_cx - half_roi)
    roi_r_y1 = max(0, img_cy - half_roi)
    roi_r_x2 = min(img_w, img_cx + half_roi)
    roi_r_y2 = min(img_h, img_cy + half_roi)
    roi_r_gray = cached_gray[roi_r_y1:roi_r_y2, roi_r_x1:roi_r_x2]
    effective_bound = min(roi_rotation, min(img_cx, img_cy))

    blurred = cv2.medianBlur(roi_r_gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=2, minDist=int(nanowell_r * 2),
        param1=70, param2=30, minRadius=int(nanowell_r * 0.85), maxRadius=int(nanowell_r * 1.15)
    )

    if circles is None or len(circles[0]) < 100:
        return None, f"Insufficient circular nodes ({0 if circles is None else len(circles[0])}/100 needed)."

    centers = circles[0][:, :2]
    dominant_angles = []
    centers_sorted = centers[centers[:, 1].argsort()]

    for _ in range(2000):
        p1 = centers_sorted[np.random.randint(0, len(centers))]
        p2 = centers_sorted[np.random.randint(0, len(centers))]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if math.sqrt(dx * dx + dy * dy) < effective_bound // 5:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        temp_angle = angle % 60.0
        if temp_angle > 30.0:
            temp_angle -= 60.0
        dominant_angles.append(temp_angle)

    if not dominant_angles:
        return None, "No dominant angle could be verified from pair distances."

    counts, bin_edges = np.histogram(dominant_angles, bins=1200, range=(-30, 30))
    peak_idx = np.argmax(counts)
    refined_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
    return refined_angle, None


def generate_grid_overlay(cached_bgr: np.ndarray, cx: int, cy: int, angle_val: float,
                          bound_r: float, pitch: float, nanowell_r: int):
    """
    Computes valid hexagonal grid points and renders the visual preview overlay.
    Returns (valid_wells, render_img).
    """
    render_img = cached_bgr.copy()
    row_spacing = pitch * math.sin(math.pi / 3.0)
    max_rows = int(bound_r / row_spacing) + 2
    max_cols = int(bound_r / pitch) + 2

    theta = math.radians(angle_val)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    valid_wells = []

    for row in range(-max_rows, max_rows + 1):
        local_y = row * row_spacing
        is_odd_row = (row % 2 != 0)
        x_offset = (pitch / 2.0) if is_odd_row else 0.0

        for col in range(-max_cols, max_cols + 1):
            local_x = col * pitch + x_offset

            if local_x ** 2 + local_y ** 2 <= bound_r ** 2:
                rot_x = local_x * cos_t - local_y * sin_t
                rot_y = local_x * sin_t + local_y * cos_t
                gx = int(cx + rot_x)
                gy = int(cy + rot_y)

                update_col = col + 1 if is_odd_row and col >= 0 else col
                valid_wells.append((gx, gy, row, update_col))
                cv2.circle(render_img, (gx, gy), nanowell_r, (0, 0, 255), 25)

    cv2.circle(render_img, (cx, cy), int(bound_r), (255, 0, 0), 30)
    cv2.circle(render_img, (cx, cy), nanowell_r, (0, 255, 0), 25)
    return valid_wells, render_img


def execute_nanowell_crop(load_path: str, well_name: str, time: str, nanowell_r: int,
                          output_size: int, valid_wells: list, log_callback=print):
    """
    Crops and exports masked single-well images across all matched channels.
    """
    load_path = load_path.strip().replace('\\', '/')
    save_base_path = os.path.join(Path(load_path).parent, "Processed Wells", well_name)
    Path(save_base_path).mkdir(parents=True, exist_ok=True)
    half_size = output_size // 2

    files = os.listdir(load_path)
    pattern = rf"^{well_name}_Time{time}_(.*)\.tif$"
    matched_channels = {}

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            channel = match.group(1)
            matched_channels[channel] = os.path.join(load_path, filename)
            Path(os.path.join(save_base_path, channel)).mkdir(parents=True, exist_ok=True)

    if not matched_channels:
        log_callback("❌ [ERROR]: Could not find any standardized channel image sets. Crop terminated.")
        return

    digital_mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.circle(digital_mask, (half_size, half_size), nanowell_r, (255, 255, 255), -1)

    for key_channel, img_path in matched_channels.items():
        log_callback(f"[BATCHING]: Crop executing for channel [{key_channel}]...")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_h, img_w = image.shape[:2]
        skip_count = 0

        for (cx, cy, row, col) in valid_wells:
            y1 = int(cy) - nanowell_r
            y2 = y1 + output_size
            x1 = int(cx) - nanowell_r
            x2 = x1 + output_size

            if y1 < 0 or x1 < 0 or y2 > img_h or x2 > img_w:
                skip_count += 1
                continue

            square_crop = image[y1:y2, x1:x2]
            cropped_with_mask = cv2.bitwise_and(square_crop, square_crop, mask=digital_mask)

            new_filename = f"{well_name}_R{row}_C{col}_Time{time}_{key_channel}.png"
            new_folder = os.path.join(save_base_path, key_channel)
            save_path = os.path.join(new_folder, new_filename)
            cv2.imwrite(save_path, cropped_with_mask)

        log_callback(f"-> Channel [{key_channel}] extracted. Skipped {skip_count} boundary nodes.")

    log_callback(f"[COMPLETE 🏁]: All channels cropped. Destination root:\n{save_base_path}")


def execute_rollback(load_path: str, well_name: str, time: str, log_callback=print):
    """
    Safely purges exported single-well crops matching criteria from disk.
    """
    load_path = load_path.strip().replace('\\', '/')
    save_base_path = os.path.join(Path(load_path).parent, "Processed Wells", well_name)

    if not os.path.exists(save_base_path):
        log_callback(f"❌ [ROLLBACK INFO]: Destination does not exist. No files purged:\n{save_base_path}")
        return

    purged_file_count = 0
    purged_folder_count = 0

    for channel_item in os.listdir(save_base_path):
        channel_dir = os.path.join(save_base_path, channel_item)
        if not os.path.isdir(channel_dir) or channel_item not in ("BF", "RGB", "mCherry", "GFP"):
            continue

        for img_name in os.listdir(channel_dir):
            match_check = re.match(rf"{well_name}_(.*)_Time{time}_(.*)\.png$", img_name)
            if match_check:
                os.remove(os.path.join(channel_dir, img_name))
                purged_file_count += 1

        if len(os.listdir(channel_dir)) == 0:
            os.rmdir(channel_dir)
            purged_folder_count += 1

    if purged_file_count > 0:
        log_callback(f"[ROLLBACK COMPLETE 🏁]: Purged {purged_file_count} files and {purged_folder_count} empty folders.")
    else:
        log_callback(f"⚠️ [ROLLBACK WARNING]: No image files matched template: '{well_name}_R*_C*_Time{time}.png'")