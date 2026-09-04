import numpy as np
import pandas as pd

from pathlib import Path

from skimage import io, color
from skimage.feature import blob_log

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def convert_to_grayscale(image):
    """
    Convert an image to a 2D grayscale numpy array.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        2D grayscale image.
    """

    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:

        # RGB/RGBA
        if image.shape[2] >= 3:
            return color.rgb2gray(image[..., :3])

        return image[..., 0]

    raise ValueError(
        f"Unsupported image dimensions: {image.shape}"
    )


def load_mcherry_crop_and_mask(crop_path, mask_dir):
    """
    Load an mCherry crop and its corresponding cell mask.

    The mask is matched using the filename stem.

    Example:
        crop: A01_cell01.png
        mask: A01_cell01.png

    Parameters
    ----------
    crop_path : str or pathlib.Path
        Path to mCherry crop.

    mask_dir : str or pathlib.Path
        Directory containing mask PNGs.

    Returns
    -------
    image : numpy.ndarray
        Grayscale mCherry image.

    mask : numpy.ndarray
        Boolean cell mask.

    mask_path : pathlib.Path
        Path to matched mask.
    """

    crop_path = Path(crop_path)
    mask_dir = Path(mask_dir)

    mask_stem = crop_path.stem.removesuffix("_mCherry")
    mask_path = mask_dir / f"{mask_stem}_BF_mask.png"

    if not mask_path.exists():
        raise FileNotFoundError(
            f"No corresponding mask found for "
            f"{crop_path.name}: {mask_path}"
        )

    image = io.imread(str(crop_path))
    mask = io.imread(str(mask_path))

    image = convert_to_grayscale(image)

    if mask.ndim == 3:
        mask = convert_to_grayscale(mask)

    mask = mask > 0

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch for {crop_path.name}: "
            f"{image.shape[:2]} vs {mask.shape[:2]}"
        )

    return image, mask, mask_path


def detect_mcherry_puncta(
    image,
    mask,
    min_diameter=2.6,
    max_diameter=8.0,
    threshold=0.09,
    num_sigma=10
):
    """
    Detect mCherry puncta using skimage.feature.blob_log.

    Only puncta whose center falls inside the supplied cell mask
    are retained.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale mCherry image.

    mask : numpy.ndarray
        Boolean cell mask.

    min_diameter : float
        Minimum punctum diameter in pixels.

    max_diameter : float
        Maximum punctum diameter in pixels.

    threshold : float
        blob_log detection threshold.

    num_sigma : int
        Number of intermediate sigma values sampled by blob_log.

    Returns
    -------
    list of dict
        Detected puncta. Each dictionary contains:

        x
        y
        sigma
        radius
        diameter
    """

    if min_diameter <= 0:
        raise ValueError(
            "min_diameter must be greater than zero."
        )

    if max_diameter < min_diameter:
        raise ValueError(
            "max_diameter must be greater than or equal "
            "to min_diameter."
        )

    if threshold <= 0:
        raise ValueError(
            "threshold must be greater than zero."
        )

    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)

    # Normalize image to 0-1 for blob detection.
    image_float = image.astype(np.float32)

    img_min = np.min(image_float)
    img_max = np.max(image_float)

    if img_max > img_min:
        image_norm = (
            image_float - img_min
        ) / (
            img_max - img_min
        )
    else:
        image_norm = np.zeros_like(image_float)

    # blob_log uses sigma rather than diameter.
    #
    # For LoG:
    #
    # diameter = 2 * sqrt(2) * sigma
    #
    min_sigma = (
        min_diameter /
        (2.0 * np.sqrt(2.0))
    )

    max_sigma = (
        max_diameter /
        (2.0 * np.sqrt(2.0))
    )

    blobs = blob_log(
        image_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )

    valid_blobs = []

    height, width = image.shape[:2]

    for blob in blobs:

        y, x, sigma = blob

        x_int = int(round(x))
        y_int = int(round(y))

        # Ignore coordinates outside the image.
        if (
            x_int < 0 or
            x_int >= width or
            y_int < 0 or
            y_int >= height
        ):
            continue

        # Only retain puncta whose CENTER is inside
        # the cell mask.
        if not mask[y_int, x_int]:
            continue

        radius = np.sqrt(2.0) * sigma

        valid_blobs.append({
            "x": float(x),
            "y": float(y),
            "sigma": float(sigma),
            "radius": float(radius),
            "diameter": float(2.0 * radius)
        })

    return valid_blobs

def plot_mcherry_analysis(
    ax,
    image,
    mask,
    puncta,
    title=None
):
    ax.clear()

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.7
    )

    for punctum in puncta:
        circle = Circle(
            (
                punctum["x"],
                punctum["y"]
            ),
            punctum["radius"],
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )
        ax.add_patch(circle)

    if title:
        ax.set_title(title)

    ax.axis("off")


def calculate_puncta_statistics(image, puncta):
    """
    Calculate mean puncta intensity and mean puncta size.

    Intensity is calculated as the mean pixel intensity inside
    each detected punctum's circular region.

    Size is the detected punctum diameter in pixels.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale mCherry image.

    puncta : list of dict
        Output from detect_mcherry_puncta().

    Returns
    -------
    mean_intensity : float
    mean_size : float
    """

    image_float = image.astype(np.float32)

    height, width = image.shape[:2]

    intensities = []
    sizes = []

    for punctum in puncta:

        x = punctum["x"]
        y = punctum["y"]
        radius = punctum["radius"]

        x_min = max(
            0,
            int(np.floor(x - radius))
        )

        x_max = min(
            width - 1,
            int(np.ceil(x + radius))
        )

        y_min = max(
            0,
            int(np.floor(y - radius))
        )

        y_max = min(
            height - 1,
            int(np.ceil(y + radius))
        )

        yy, xx = np.ogrid[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

        circular_mask = (
            (xx - x) ** 2 +
            (yy - y) ** 2
        ) <= radius ** 2

        pixels = image_float[
            y_min:y_max + 1,
            x_min:x_max + 1
        ][circular_mask]

        if pixels.size > 0:
            intensities.append(
                float(np.mean(pixels))
            )

        sizes.append(
            punctum["diameter"]
        )

    if intensities:
        mean_intensity = float(
            np.mean(intensities)
        )
    else:
        mean_intensity = 0.0

    if sizes:
        mean_size = float(
            np.mean(sizes)
        )
    else:
        mean_size = 0.0

    return mean_intensity, mean_size


def create_mcherry_figure(
    image,
    mask,
    puncta,
    title=None,
    figure_size=(7, 6)
):
    """
    Create a matplotlib visualization of mCherry puncta.

    The mCherry image is shown in grayscale.
    The cell mask boundary is shown in green.
    Detected puncta are shown as red outlines.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale mCherry image.

    mask : numpy.ndarray
        Boolean cell mask.

    puncta : list of dict
        Detected puncta.

    title : str, optional
        Plot title.

    figure_size : tuple
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    # Cell mask boundary.
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.7,
        alpha=0.8
    )

    # Puncta.
    for punctum in puncta:

        circle = Circle(
            (
                punctum["x"],
                punctum["y"]
            ),
            punctum["radius"],
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )

        ax.add_patch(circle)

    if title is not None:
        ax.set_title(
            title,
            fontsize=11
        )

    ax.axis("off")

    fig.tight_layout()

    return fig


def analyze_mcherry_image(
    crop_path,
    mask_dir,
    min_diameter=2.6,
    max_diameter=8.0,
    threshold=0.09
):
    """
    Complete analysis pipeline for one mCherry crop.

    Returns the loaded image, mask, puncta, and statistics.
    """

    image, mask, mask_path = (
        load_mcherry_crop_and_mask(
            crop_path,
            mask_dir
        )
    )

    puncta = detect_mcherry_puncta(
        image=image,
        mask=mask,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        threshold=threshold
    )

    (
        mean_intensity,
        mean_size
    ) = calculate_puncta_statistics(
        image,
        puncta
    )

    return {
        "image": image,
        "mask": mask,
        "mask_path": mask_path,
        "puncta": puncta,
        "count": len(puncta),
        "mean_intensity": mean_intensity,
        "mean_size": mean_size
    }


def save_mcherry_figure(
    fig,
    output_path,
    dpi=300
):
    """
    Save an mCherry visualization.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        str(output_path),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


def batch_analyze_mcherry(
    crop_dir,
    mask_dir,
    output_dir,
    min_diameter=2.6,
    max_diameter=8.0,
    threshold=0.09,
    progress_callback=None,
    log_callback=None
):
    """
    Batch process all mCherry PNGs.

    Parameters
    ----------
    crop_dir : str or Path
        Directory containing mCherry crop PNGs.

    mask_dir : str or Path
        Directory containing corresponding masks.

    output_dir : str or Path
        Output directory.

    min_diameter : float
        Minimum punctum diameter.

    max_diameter : float
        Maximum punctum diameter.

    threshold : float
        blob_log threshold.

    progress_callback : callable, optional
        Function receiving integer progress 0-100.

    log_callback : callable, optional
        Function receiving log strings.

    Returns
    -------
    pandas.DataFrame
        Batch results.
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
        crop_dir.glob("*.png")
    )

    if not crop_files:
        raise FileNotFoundError(
            f"No PNG files found in {crop_dir}"
        )

    results = []

    total = len(crop_files)

    for index, crop_path in enumerate(crop_files):

        try:

            analysis = analyze_mcherry_image(
                crop_path=crop_path,
                mask_dir=mask_dir,
                min_diameter=min_diameter,
                max_diameter=max_diameter,
                threshold=threshold
            )

            puncta = analysis["puncta"]

            title = (
                f"{crop_path.name}\n"
                f"Puncta: {analysis['count']} | "
                f"Mean Intensity: "
                f"{analysis['mean_intensity']:.2f} | "
                f"Mean Diameter: "
                f"{analysis['mean_size']:.2f} px"
            )

            fig = create_mcherry_figure(
                image=analysis["image"],
                mask=analysis["mask"],
                puncta=puncta,
                title=title
            )

            output_path = (
                visualization_dir /
                f"{crop_path.stem}_mCherry_puncta.png"
            )

            save_mcherry_figure(
                fig,
                output_path
            )

            results.append({
                "image": crop_path.name,
                "number_of_mCherry_puncta":
                    analysis["count"],
                "mean_intensity_of_puncta":
                    analysis["mean_intensity"],
                "mean_size_of_puncta":
                    analysis["mean_size"],
                "min_diameter":
                    min_diameter,
                "max_diameter":
                    max_diameter,
                "threshold":
                    threshold
            })

            if log_callback:
                log_callback(
                    f"[mCherry]: {crop_path.name} → "
                    f"{analysis['count']} puncta | "
                    f"Mean intensity = "
                    f"{analysis['mean_intensity']:.2f} | "
                    f"Mean size = "
                    f"{analysis['mean_size']:.2f} px"
                )

        except Exception as e:

            if log_callback:
                log_callback(
                    f"[mCherry ERROR]: "
                    f"{crop_path.name}: {e}"
                )

            results.append({
                "image": crop_path.name,
                "number_of_mCherry_puncta": np.nan,
                "mean_intensity_of_puncta": np.nan,
                "mean_size_of_puncta": np.nan,
                "min_diameter": min_diameter,
                "max_diameter": max_diameter,
                "threshold": threshold
            })

        if progress_callback:
            progress = int(
                ((index + 1) / total) * 100
            )
            progress_callback(progress)

    results_df = pd.DataFrame(results)

    csv_path = (
        output_dir /
        "mCherry_puncta_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    if log_callback:
        log_callback(
            f"[mCherry]: CSV saved → {csv_path}"
        )

    return results_df
