"""
exr_collage.py
==============
Utility to generate PNG collages from 6 EXR cube-face images
(back, bottom, front, left, right, top).

Three collages are produced per run:
  1. RGBA channel      → <SAVE_PATH>/collage_rgba.png
  2. Depth channel     → <SAVE_PATH>/collage_depth.png   (with saturation mapping)
  3. N (normal) channel→ <SAVE_PATH>/collage_normals.png

Layout – 3 columns × 2 rows, faces in the order supplied:
  [0] [1] [2]
  [3] [4] [5]

If the source images are 1024 × 1024, the output canvas is 3072 × 2048.

Dependencies
------------
  pip install OpenEXR Imath Pillow numpy

Usage (CLI)
-----------
  python exr_collage.py back.exr bottom.exr front.exr left.exr right.exr top.exr
  python exr_collage.py back.exr bottom.exr front.exr left.exr right.exr top.exr \
      --save-path ./out --resize-to 512 --depth-saturation 0.75
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import OpenEXR
import Imath
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# ── Global configuration ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

SAVE_PATH: str          = "./output"   # Directory where collages are saved
DEPTH_SATURATION: float = 0.82         # Saturation for depth colour ramp (0–1)
RESIZE_TO: Optional[int] = None        # Resize each face to N×N before tiling
                                       # (None = keep original size)

# ---------------------------------------------------------------------------
# ── Internal EXR helpers ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def _open_exr(path: str) -> OpenEXR.InputFile:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path}")
    return OpenEXR.InputFile(path)


def _exr_size(exr: OpenEXR.InputFile) -> Tuple[int, int]:
    dw     = exr.header()["dataWindow"]
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    return width, height


def _available_channels(exr: OpenEXR.InputFile) -> List[str]:
    return list(exr.header()["channels"].keys())


def _find_channel(channels: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate present in channels (case-insensitive)."""
    lower_map = {c.lower(): c for c in channels}
    for cand in candidates:
        found = lower_map.get(cand.lower())
        if found is not None:
            return found
    return None


def _read_channel(exr: OpenEXR.InputFile,
                  name: str,
                  width: int,
                  height: int) -> np.ndarray:
    raw = exr.channel(name, _FLOAT)
    return np.frombuffer(raw, dtype=np.float32).reshape(height, width)


# ---------------------------------------------------------------------------
# ── Channel extractors ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def extract_rgba(exr: OpenEXR.InputFile) -> np.ndarray:
    """
    Return an (H, W, 4) uint8 RGBA array.
    Falls back to alpha=1 when no A channel is present.
    """
    w, h   = _exr_size(exr)
    avail  = _available_channels(exr)

    r_name = _find_channel(avail, ["R", "r", "red"])
    g_name = _find_channel(avail, ["G", "g", "green"])
    b_name = _find_channel(avail, ["B", "b", "blue"])
    a_name = _find_channel(avail, ["A", "a", "Alpha", "alpha"])

    if None in (r_name, g_name, b_name):
        raise KeyError(
            f"Could not find R/G/B channels.  Available: {avail}"
        )

    r = _read_channel(exr, r_name, w, h)
    g = _read_channel(exr, g_name, w, h)
    b = _read_channel(exr, b_name, w, h)
    a = _read_channel(exr, a_name, w, h) if a_name else np.ones((h, w), np.float32)

    rgba = np.stack([r, g, b, a], axis=-1)
    rgba = np.clip(rgba, 0.0, 1.0)
    return (rgba * 255).astype(np.uint8)


def extract_depth(exr: OpenEXR.InputFile,
                  saturation: float = DEPTH_SATURATION) -> np.ndarray:
    """
    Return an (H, W, 3) uint8 RGB array from the depth channel.

    Robust handling
    ---------------
    Many renderers store a large sentinel value (e.g. 1e5, 1e9) in the Z
    channel for background / sky pixels that were never hit by a ray.
    A naïve min/max normalisation then crushes the entire foreground into
    near-black because the sentinel dominates the range.

    This function:
      1. Detects background pixels via a gap heuristic – any pixel whose
         Z value is > ``DEPTH_SENTINEL_THRESHOLD`` times the 95th-percentile
         of the full image is treated as background.
      2. Normalises only the foreground pixels, clipping between the 1st
         and 99th percentile to discard outliers.
      3. Inverts the result so that near geometry = bright, far = dark.
      4. Sets background pixels to 0 (black) in the output.
      5. Maps normalised depth through an HSV colour ramp (blue→red) with
         the given saturation.  Set saturation=0 to get a plain greyscale map.

    Parameters
    ----------
    exr : OpenEXR.InputFile
    saturation : float
        HSV saturation applied to the colour ramp (0 = greyscale, 1 = vivid).
    """
    w, h  = _exr_size(exr)
    avail = _available_channels(exr)

    d_name = _find_channel(
        avail,
        ["Z", "z", "depth", "Depth", "P.Z", "P.z", "position.Z", "position.z"]
    )
    if d_name is None:
        raise KeyError(
            f"No depth channel found.  Available channels: {avail}"
        )

    depth = _read_channel(exr, d_name, w, h).astype(np.float64)

    # ── 1. Detect background sentinel pixels ────────────────────────────
    # Use a gap heuristic: compute the 95th percentile of the whole image.
    # Pixels that are more than SENTINEL_RATIO × p95 are background.
    SENTINEL_RATIO = 5.0
    p95 = float(np.percentile(depth, 95))
    bg_mask = depth > (p95 * SENTINEL_RATIO)   # True = background / no-hit

    fg_pixels = depth[~bg_mask]

    if fg_pixels.size == 0:
        # Entire image is background – return solid black
        return np.zeros((h, w, 3), dtype=np.uint8)

    # ── 2. Normalise foreground only (percentile-clipped) ───────────────
    z_min = float(np.percentile(fg_pixels, 1))
    z_max = float(np.percentile(fg_pixels, 99))

    if z_max > z_min:
        normed = np.clip((depth - z_min) / (z_max - z_min), 0.0, 1.0)
    else:
        normed = np.zeros_like(depth)

    # ── 3. Invert: near = 1.0 (bright/white), far = 0.0 (dark) ─────────
    normed = 1.0 - normed

    # ── 4. Zero-out background pixels ───────────────────────────────────
    normed[bg_mask] = 0.0

    # ── 5. Vectorised HSV → RGB colour ramp ─────────────────────────────
    #       hue sweeps 0.6 (blue) at near → 0.0 (red) at far
    #       saturation=0 produces a clean greyscale depth map
    if saturation < 1e-6:
        # Fast path: pure greyscale
        grey = (normed * 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)

    hue   = 0.6 * normed                        # 0.0 (dark/far) … 0.6 (bright/near)
    sat   = np.full_like(normed, saturation)
    value = normed

    i = (hue * 6).astype(np.int32) % 6
    f = hue * 6 - np.floor(hue * 6)
    p = value * (1.0 - sat)
    q = value * (1.0 - f * sat)
    t = value * (1.0 - (1.0 - f) * sat)

    r_ch = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [value, q, p, p, t, value])
    g_ch = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, value, value, q, p, p])
    b_ch = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, value, value, q])

    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def extract_normals(exr: OpenEXR.InputFile) -> np.ndarray:
    """
    Return an (H, W, 3) uint8 RGB array from the N (normal) channel.

    XYZ components are remapped from [-1, 1] → [0, 255].
    Tries common naming conventions (N.X/Y/Z, normal.X/Y/Z, nx/ny/nz, …).
    """
    w, h  = _exr_size(exr)
    avail = _available_channels(exr)

    x_name = _find_channel(avail, ["N.X", "N.x", "normal.X", "normal.x", "nx", "Nx", "NX"])
    y_name = _find_channel(avail, ["N.Y", "N.y", "normal.Y", "normal.y", "ny", "Ny", "NY"])
    z_name = _find_channel(avail, ["N.Z", "N.z", "normal.Z", "normal.z", "nz", "Nz", "NZ"])

    if None in (x_name, y_name, z_name):
        missing = [ax for ax, n in zip("XYZ", [x_name, y_name, z_name]) if n is None]
        raise KeyError(
            f"Could not locate normal channel(s) for axis: {missing}.  "
            f"Available: {avail}"
        )

    nx = _read_channel(exr, x_name, w, h)
    ny = _read_channel(exr, y_name, w, h)
    nz = _read_channel(exr, z_name, w, h)

    # [-1, 1] → [0, 1]
    nx = nx * 0.5 + 0.5
    ny = ny * 0.5 + 0.5
    nz = nz * 0.5 + 0.5

    rgb = np.stack([nx, ny, nz], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ── Collage utilities ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def build_collage(images: List[np.ndarray]) -> np.ndarray:
    """
    Tile 6 images into a 3-column × 2-row grid.

    All images must share the same (H, W).  The number of colour
    channels (C) is taken from the first image.

    Returns
    -------
    np.ndarray of shape (H*2, W*3, C) uint8.
    """
    if len(images) != 6:
        raise ValueError(f"Expected exactly 6 images, got {len(images)}")

    h, w = images[0].shape[:2]
    c    = images[0].shape[2] if images[0].ndim == 3 else 1

    canvas = np.zeros((h * 2, w * 3, c), dtype=np.uint8)
    for idx, img in enumerate(images):
        row = idx // 3
        col = idx %  3
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        canvas[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    return canvas


def resize_face(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a uint8 face image to (size, size) using Lanczos resampling."""
    c   = arr.shape[2] if arr.ndim == 3 else 1
    mode = "RGBA" if c == 4 else "RGB"
    pil  = PILImage.fromarray(arr, mode)
    pil  = pil.resize((size, size), PILImage.LANCZOS)
    return np.asarray(pil)


def save_png(canvas: np.ndarray, path: str) -> None:
    """Save a uint8 numpy array as a PNG file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c    = canvas.shape[2] if canvas.ndim == 3 else 1
    mode = "RGBA" if c == 4 else "RGB"
    PILImage.fromarray(canvas.squeeze(), mode).save(path, format="PNG")
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# ── Main class ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class EXRCollageGenerator:
    """
    Generate three PNG collages (RGBA / Depth / Normals) from 6 EXR face images.

    Parameters
    ----------
    image_paths : list[str]
        Exactly 6 paths in the order: back, bottom, front, left, right, top.
    save_path : str
        Output directory (created if absent).
        Default: module-level ``SAVE_PATH``.
    depth_saturation : float
        Saturation for the depth colour ramp (0 = greyscale, 1 = full colour).
        Default: module-level ``DEPTH_SATURATION``.
    resize_to : int | None
        When set, each face is resized to ``resize_to × resize_to`` pixels
        before compositing.
        Default: module-level ``RESIZE_TO`` (None → keep original size).
    """

    FACE_ORDER = ["back", "bottom", "front", "left", "right", "top"]

    def __init__(
        self,
        image_paths:      List[str],
        save_path:        str            = SAVE_PATH,
        depth_saturation: float          = DEPTH_SATURATION,
        resize_to:        Optional[int]  = RESIZE_TO,
    ) -> None:
        if len(image_paths) != 6:
            raise ValueError(
                f"Exactly 6 image paths required "
                f"(back / bottom / front / left / right / top); "
                f"received {len(image_paths)}."
            )
        self.image_paths      = image_paths
        self.save_path        = save_path
        self.depth_saturation = depth_saturation
        self.resize_to        = resize_to

    # ------------------------------------------------------------------
    def _extract_all(self,
                     extractor: Callable[[OpenEXR.InputFile], np.ndarray],
                     label: str) -> List[np.ndarray]:
        """Open all 6 EXR files and apply *extractor* to each."""
        results: List[np.ndarray] = []
        for i, path in enumerate(self.image_paths):
            face = self.FACE_ORDER[i]
            print(f"    [{face:6s}] {path}")
            exr = _open_exr(path)
            try:
                arr = extractor(exr)
            finally:
                exr.close()

            if self.resize_to is not None:
                arr = resize_face(arr, self.resize_to)

            results.append(arr)
        return results

    # ------------------------------------------------------------------
    def generate_rgba_collage(self) -> str:
        """Extract RGBA, build 3×2 collage, save PNG.  Returns output path."""
        print("[RGBA] Extracting …")
        images   = self._extract_all(extract_rgba, "RGBA")
        canvas   = build_collage(images)
        out_path = os.path.join(self.save_path, "collage_rgba.png")
        save_png(canvas, out_path)
        return out_path

    # ------------------------------------------------------------------
    def generate_depth_collage(self) -> str:
        """Extract depth, build 3×2 collage, save PNG.  Returns output path."""
        print(f"[Depth] Extracting (saturation={self.depth_saturation}) …")

        def _extractor(exr: OpenEXR.InputFile) -> np.ndarray:
            return extract_depth(exr, saturation=self.depth_saturation)

        images   = self._extract_all(_extractor, "Depth")
        canvas   = build_collage(images)
        out_path = os.path.join(self.save_path, "collage_depth.png")
        save_png(canvas, out_path)
        return out_path

    # ------------------------------------------------------------------
    def generate_normals_collage(self) -> str:
        """Extract normals, build 3×2 collage, save PNG.  Returns output path."""
        print("[N] Extracting normal channels …")
        images   = self._extract_all(extract_normals, "N")
        canvas   = build_collage(images)
        out_path = os.path.join(self.save_path, "collage_normals.png")
        save_png(canvas, out_path)
        return out_path

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run all three collage extractions and return a dict with the
        output file paths keyed by ``'rgba'``, ``'depth'``, ``'normals'``.
        """
        print("=" * 62)
        print("  EXR Collage Generator")
        print(f"  Face order   : {', '.join(self.FACE_ORDER)}")
        print(f"  Save path    : {self.save_path}")
        print(f"  Resize to    : {self.resize_to if self.resize_to else 'original size'}")
        print(f"  Depth sat.   : {self.depth_saturation}")
        print("=" * 62)

        return {
            "rgba":    self.generate_rgba_collage(),
            "depth":   self.generate_depth_collage(),
            "normals": self.generate_normals_collage(),
        }


if __name__ == "__main__":
    images_searchPath = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\temp"
    FACE_ORDER = ["front", "left", "top", "back", "right", "bottom"]
    images = [os.path.join(images_searchPath, f"{face}.exr") for face in FACE_ORDER]
    
    gen = EXRCollageGenerator(
        image_paths      = images,
        save_path        = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\output",
        depth_saturation = 0.5,
        resize_to        = 1024,
    )

    outputs = gen.run()

    print("\nOutput files:")
    for channel, path in outputs.items():
        print(f"  {channel:8s} → {path}")