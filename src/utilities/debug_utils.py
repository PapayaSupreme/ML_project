from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def load_debug_tiles(debug_dir: str | Path | None = None, *, normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Load extracted debug tile PNGs from the vectorized_output debug folder.

    Returns:
        X: np.ndarray shape (n_samples, 784) dtype float32 if normalize else uint8
        filenames: list of file names (str) in the same order as rows in X

    The function looks for files matching "*_tile_*.png" in the debug dir.
    """
    if debug_dir is None:
        # default debug directory used by vectorize_images
        debug_dir = Path(__file__).resolve().parents[1] / "data" / "vectorized_output" / "debug"
    else:
        debug_dir = Path(debug_dir)

    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug directory not found: {debug_dir}")

    png_files = sorted([p for p in debug_dir.glob("*.png") if "_tile_" in p.name])
    if not png_files:
        raise FileNotFoundError(f"No debug tile PNGs found in: {debug_dir}")

    images: list[np.ndarray] = []
    names: list[str] = []

    for p in png_files:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("L"))
        images.append(arr.reshape(-1))
        names.append(p.name)

    X = np.stack(images, axis=0)
    if normalize:
        X = X.astype(np.float32) / 255.0
    return X, names

