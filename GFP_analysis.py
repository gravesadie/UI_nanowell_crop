from matplotlib import image
import numpy as np
import pandas as pd

from pathlib import Path

from skimage import io, color
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from mcherry_analysis import load_mcherry_crop_and_mask

# NEEDS UPDATING TO MATCH MULTI-CELL FIXES

# also generally:
# normalize background signal in different wells
# subtract nanowell artifacts using pre-seeded well images? capture in GFP channel?

def analyze_GFP_image(
    mask_path,
    crop_dir):

    image, mask, mask_path = (
        load_mcherry_crop_and_mask(
            mask_path,
            crop_dir,
            channel='GFP'))

    analysis = {
        "image": image,
        "mask": mask,
        "image_path": mask_path.name,
        "I_sum": np.sum(image * mask),
        "I_max": np.max(image * mask),
        "I_mean": np.mean(image * mask),
        "I_sd": np.std(image * mask)
    }
    return analysis

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
            analysis = analyze_GFP_image(
                mask_path=mask_path,
                crop_dir=crop_dir
            )

            if not(analysis):
                continue

            fig = create_GFP_figure(
                image=analysis['image'],
                mask=analysis['mask'],
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

            results.append({
                "image": mask_path_smpl.name,
                "I_sum":  analysis["I_sum"],
                "I_max":  analysis["I_max"],
                "I_mean": analysis["I_mean"],
                "I_sd": analysis["I_sd"]})

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
            f"[GFP]: CSV saved → {csv_path}"
        )

    return results_df