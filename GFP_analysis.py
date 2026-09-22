import numpy as np
import pandas as pd

from pathlib import Path

from skimage import io, color
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from mcherry_analysis_v2_multicell import load_mcherry_crop_and_mask

def measure_intensity(region, image, mask):
    '''Quantify the intensity of GFP expression using cell mask as bound'''
    area = region.area
    sum_I = np.sum(image * mask)
    I_by_area = sum_I / area
    stats = (area, sum_I, I_by_area, region.centroid[0],region.centroid[1], region.intensity_max, region.intensity_mean, region.intensity_std)
    return stats

def analyze_GFP_image(mask_path, crop_dir, results):

    image, mask, mask_path = (
        load_mcherry_crop_and_mask(
            mask_path,
            crop_dir,
            channel='GFP'))

    regions = regionprops(label(mask), image)
    # iterate through multiple cells in same well if present
    if len(regions) == 0:
        return False
    for (i, region) in enumerate(regions):
        mask_i = mask == (i+1)
        int_stats = measure_intensity(region, image, mask_i)

        res = {
            "image": mask_path.name,
            "cell_i": i,
            "ID": f"{mask_path.stem}_cell{i}",
            "area": int_stats[0],
            "centroid_x": int_stats[3],
            "centroid_y": int_stats[4],
            "GFP_I_sum": int_stats[1],
            "GFP_I_by_area": int_stats[2],
            "GFP_I_max": int_stats[5],
            "GFP_I_mean": int_stats[6],
            "GFP_I_sd": int_stats[7]
        }
        results.append(res)
    return (image, mask)

def create_GFP_figure(
    image,
    mask,
    title=None,
    figure_size=(7, 6)
):
    fig, ax = plt.subplots(
        figsize=figure_size
    )

    ax.imshow(
        image,
        cmap="gray")

    # Cell mask boundary.
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.7,
        alpha=0.8
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=11
        )
    ax.axis("off")
    fig.tight_layout()
    return fig

def batch_analyze_GFP(
    crop_dir,
    mask_dir,
    output_dir,
    progress_callback=None,
    log_callback=None
):
    """
    See mCherry equivalent
    """
    crop_dir = Path(crop_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    visualization_dir = (
        output_dir / "Visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    crop_files = sorted(
        crop_dir.glob("*_GFP.png"))
    mask_files = sorted(
        mask_dir.glob("*_mask.png"))

    if not crop_files:
        raise FileNotFoundError(
            f"No PNG files found in {crop_dir}"
        )

    results = []

    total = len(mask_files)

    for index, mask_path in enumerate(mask_files):
        mask_path_smpl = Path("_".join(mask_path.stem.split("_")[:-2]))
        try:
            img_mask_zip = analyze_GFP_image(
                mask_path=mask_path,
                crop_dir=crop_dir,
                results=results
            )

            #if not(img_mask_zip):
            #    continue

            fig = create_GFP_figure(
                image=img_mask_zip[0],
                mask=img_mask_zip[1],
                title=f"{mask_path_smpl.name}\n"
            )

            output_path = (
                visualization_dir /
                f"{mask_path_smpl.stem}_GFP_contour.png"
            )

            fig.savefig(
                str(output_path),
                bbox_inches="tight",
                facecolor="white"
            )
            plt.close(fig)

            if log_callback:
                log_callback(
                    f"[GFP]: {mask_path_smpl.name}"
                )

        except Exception as e:

            if log_callback:
                log_callback(
                    f"[GFP ERROR]: "
                    f"{mask_path.name}: {e}"
                )

        if progress_callback:
            progress = int(
                ((index + 1) / total) * 100
            )
            progress_callback(progress)

    results_df = pd.DataFrame(results)

    csv_path = (
        output_dir /
        "GFP_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    if log_callback:
        log_callback(
            f"[GFP]: CSV saved → {csv_path} \n"
            f"{len(results)} cells and {total} images/wells processed."
        )

    return results_df