from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import numpy as np
from PIL import Image

_LABEL_PATTERN = re.compile(r"MNIST-(\d)")


def load_vectorized_digits(
    data_dir: str | Path | None = None,
    *,
    tile_size: Tuple[int, int] = (28, 28),
    grid_size: Tuple[int, int] = (25, 40),
    normalize: bool = False,
    crop_box: tuple[int, int, int, int] = (295, 200, 1720, 960), #left top right bottom
    output_dir: str | Path | None = None,
    save_vectors: bool = True,
    save_debug: bool = True,
    debug_tiles_per_file: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Load rendered MNIST grid PNG files into vectorized samples and labels.

    The input PNGs are expected to be rendered 25x40 digit sheets like:
    ``mnist_v5_MNIST-9_01001-02000_25x40.png``.

    Because those PNGs contain margins, axis labels, and a filename caption, this
    function first crops the digit area, then splits that crop into 25x40 tiles,
    resizes each tile to 28x28, flattens each tile into a vector of length 784,
    and optionally saves the vectors and debug images.

    Args:
        data_dir: Directory containing PNG files. Defaults to ``src/data``.
        tile_size: Final output tile size as ``(height, width)``.
        grid_size: Number of digit rows and columns in each PNG sheet.
        normalize: If True, saves/returns float32 features in [0.0, 1.0].
        crop_box: PIL-style crop box ``(left, top, right, bottom)`` selecting
            only the digit grid, excluding labels/margins/caption.
        output_dir: Directory where vectors/debug files are saved. Defaults to
            ``data_dir / "vectorized_output"``.
        save_vectors: If True, saves ``X_vectors.npy`` and ``y_labels.npy``.
        save_debug: If True, saves crop and sample tile images for inspection.
        debug_tiles_per_file: Number of extracted tiles to save per PNG file.

    Returns:
        X: Array of shape ``(n_samples, 784)``.
        y: Array of shape ``(n_samples,)``.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    else:
        data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir / "vectorized_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(data_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {data_dir}")

    tile_h, tile_w = tile_size
    grid_rows, grid_cols = grid_size
    left, top, right, bottom = crop_box

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for png_path in png_files:
        match = _LABEL_PATTERN.search(png_path.name)
        if not match:
            raise ValueError(f"Could not infer label from filename: {png_path.name}")
        label = int(match.group(1))

        with Image.open(png_path) as image:
            image_array = np.asarray(image.convert("L"), dtype=np.uint8)

        image_h, image_w = image_array.shape
        if not (0 <= left < right <= image_w and 0 <= top < bottom <= image_h):
            raise ValueError(
                f"Invalid crop_box={crop_box} for {png_path.name} with shape "
                f"height={image_h}, width={image_w}"
            )

        # Crop only the digit grid, excluding axis labels, margins, and caption.
        digit_area = image_array[top:bottom, left:right]

        if save_debug:
            Image.fromarray(digit_area).save(debug_dir / f"{png_path.stem}_crop.png")

        row_tiles = np.array_split(digit_area, grid_rows, axis=0)
        flattened_tiles: list[np.ndarray] = []
        debug_tiles: list[np.ndarray] = []

        for row_idx, row in enumerate(row_tiles):
            col_tiles = np.array_split(row, grid_cols, axis=1)
            for col_idx, tile in enumerate(col_tiles):
                resized_tile = np.asarray(
                    Image.fromarray(tile).resize((tile_w, tile_h), Image.BILINEAR),
                    dtype=np.uint8,
                )
                flattened_tiles.append(resized_tile.reshape(tile_h * tile_w))

                tile_index = row_idx * grid_cols + col_idx
                if save_debug and tile_index < debug_tiles_per_file:
                    debug_tiles.append(resized_tile)

        patches = np.stack(flattened_tiles, axis=0)

        if save_debug:
            for i, tile in enumerate(debug_tiles):
                Image.fromarray(tile).save(debug_dir / f"{png_path.stem}_tile_{i:03d}.png")

        if normalize:
            patches = patches.astype(np.float32) / 255.0

        labels = np.full((patches.shape[0],), label, dtype=np.int64)

        all_x.append(patches)
        all_y.append(labels)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    if save_vectors:
        np.save(output_dir / "X_vectors.npy", x)
        np.save(output_dir / "y_labels.npy", y)

    return x, y


if __name__ == "__main__":
    X, y = load_vectorized_digits(
        normalize=True,
        save_vectors=True,
        save_debug=True,
    )

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, labels: {np.unique(y)}")
    print("Saved vectors to: data/vectorized_output/X_vectors.npy")
    print("Saved labels to:  data/vectorized_output/y_labels.npy")
    print("Saved debug files to: data/vectorized_output/debug/")
