"""
collage_splitter.py
===================
Reverse of exr_collage.py.

Given a 3×2 PNG collage (3 columns, 2 rows), this utility:
  1. Cuts each of the 6 faces out pixel-accurately from the collage grid.
  2. Optionally resizes each face to OUTPUT_SIZE × OUTPUT_SIZE.
  3. Saves each face as  ``{material_name}.{UDIM}.png`` inside a
     per-material sub-folder created under OUTPUT_ROOT.

UDIM pipeline
-------------
UDIM tiles start at 1001 and increment left-to-right, top-to-bottom,
matching the standard Mari / Substance / Houdini convention:

  Grid position   UDIM
  ─────────────   ────
  row 0, col 0 → 1001   (face[0])
  row 0, col 1 → 1002   (face[1])
  row 0, col 2 → 1003   (face[2])
  row 1, col 0 → 1004   (face[3])
  row 1, col 1 → 1005   (face[4])
  row 1, col 2 → 1006   (face[5])

FACE_ORDER maps UDIM slot → face name.

Usage
-----
As a module::

    from collage_splitter import CollageSplitter
    s = CollageSplitter(
        collage_path  = "./output/collage_rgba.png",
        material_name = "robot_rgba",
    )
    s.run()

CLI::

    python collage_splitter.py ./output/collage_rgba.png robot_rgba
    python collage_splitter.py ./output/collage_depth.png robot_depth \\
        --output-root ./textures --output-size 512

Dependencies
------------
    pip install Pillow
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# ── Global configuration ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Order of faces in the collage grid (left-to-right, top-to-bottom).
# Edit this list to match the order used when the collage was generated.
FACE_ORDER: List[str] = ["front", "left", "top", "back", "right", "bottom"]

# Root directory under which per-material sub-folders are created.
OUTPUT_ROOT: str = "./output"

# UDIM index of the first tile (standard Mari convention).
UDIM_START: int = 1001

# If set, each extracted face is resized to OUTPUT_SIZE × OUTPUT_SIZE
# **after** the pixel-accurate cut.  None = keep the native tile size.
OUTPUT_SIZE: Optional[int] = None

# Resampling filter used for the optional resize step.
# PIL.Image.LANCZOS gives the highest quality downscale.
RESAMPLE_FILTER = PILImage.LANCZOS

# ---------------------------------------------------------------------------
# ── Core helpers ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _tile_size(collage: PILImage.Image, cols: int = 3, rows: int = 2) -> Tuple[int, int]:
    """
    Compute the pixel-accurate size of each tile in the collage.

    The collage must be exactly ``cols × tile_w`` wide and
    ``rows × tile_h`` tall.  An error is raised if the dimensions do not
    divide evenly — this would indicate the image was saved with padding
    or that the wrong collage was provided.

    Returns
    -------
    (tile_w, tile_h)
    """
    cw, ch = collage.size
    if cw % cols != 0:
        raise ValueError(
            f"Collage width {cw} is not evenly divisible by {cols} columns."
        )
    if ch % rows != 0:
        raise ValueError(
            f"Collage height {ch} is not evenly divisible by {rows} rows."
        )
    return cw // cols, ch // rows


def _crop_tile(collage: PILImage.Image,
               tile_w: int, tile_h: int,
               row: int, col: int) -> PILImage.Image:
    """
    Return a pixel-accurate crop of one tile from the collage.

    Parameters
    ----------
    collage : PIL image
    tile_w, tile_h : tile dimensions in pixels
    row, col : zero-based grid position
    """
    left   = col * tile_w
    upper  = row * tile_h
    right  = left + tile_w
    lower  = upper + tile_h
    return collage.crop((left, upper, right, lower))


def _udim(index: int, start: int = UDIM_START) -> int:
    """Return the UDIM index for the *index*-th tile (0-based)."""
    return start + index


# ---------------------------------------------------------------------------
# ── Main class ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class CollageSplitter:
    """
    Split a 3×2 PNG collage back into individual face images with UDIM names.

    Parameters
    ----------
    collage_path : str
        Path to the source collage PNG.
    material_name : str
        Base name used for the output files and the sub-folder.
        Output files will be named ``{material_name}.{UDIM}.png``.
    output_root : str
        Root directory.  A sub-folder named ``{material_name}`` is created
        inside it.  Defaults to the module-level ``OUTPUT_ROOT``.
    output_size : int | None
        If given, each tile is resized to ``output_size × output_size``
        **after** the pixel-accurate crop.  Defaults to ``OUTPUT_SIZE``.
    face_order : list[str]
        Names of the 6 faces in collage order (left→right, top→bottom).
        Defaults to the module-level ``FACE_ORDER``.
    udim_start : int
        First UDIM index.  Defaults to ``UDIM_START`` (1001).
    cols : int
        Number of columns in the collage grid (default 3).
    rows : int
        Number of rows in the collage grid (default 2).
    """

    def __init__(
        self,
        collage_path:  str,
        material_name: str,
        output_root:   str            = OUTPUT_ROOT,
        output_size:   Optional[int]  = OUTPUT_SIZE,
        face_order:    List[str]      = FACE_ORDER,
        udim_start:    int            = UDIM_START,
        cols:          int            = 3,
        rows:          int            = 2,
    ) -> None:
        if not os.path.isfile(collage_path):
            raise FileNotFoundError(f"Collage not found: {collage_path}")

        expected = cols * rows
        if len(face_order) != expected:
            raise ValueError(
                f"face_order must have exactly {expected} entries for a "
                f"{cols}×{rows} grid; got {len(face_order)}."
            )

        self.collage_path  = collage_path
        self.material_name = material_name
        self.output_root   = output_root
        self.output_size   = output_size
        self.face_order    = face_order
        self.udim_start    = udim_start
        self.cols          = cols
        self.rows          = rows

    # ------------------------------------------------------------------
    @property
    def output_folder(self) -> Path:
        """The per-material sub-folder where tiles are written."""
        return Path(self.output_root) / self.material_name

    # ------------------------------------------------------------------
    def _make_output_folder(self) -> None:
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"  Output folder : {self.output_folder.resolve()}")

    # ------------------------------------------------------------------
    def split(self) -> List[str]:
        """
        Perform the split and return a list of written file paths.

        Steps
        -----
        1. Open collage and compute tile dimensions (pixel-accurate).
        2. For each tile position (row, col):
           a. Crop the tile exactly.
           b. Optionally resize.
           c. Save as ``{material_name}.{UDIM}.png``.
        """
        self._make_output_folder()

        collage = PILImage.open(self.collage_path)
        tile_w, tile_h = _tile_size(collage, self.cols, self.rows)

        print(f"  Collage size  : {collage.size[0]} × {collage.size[1]} px")
        print(f"  Tile size     : {tile_w} × {tile_h} px (pixel-accurate)")
        if self.output_size:
            print(f"  Resize to     : {self.output_size} × {self.output_size} px")
        print()

        written: List[str] = []

        for idx, face_name in enumerate(self.face_order):
            row  = idx // self.cols
            col  = idx %  self.cols
            udim = 1001 + col + row * 10

            # ── 1. Pixel-accurate crop ───────────────────────────────
            tile = _crop_tile(collage, tile_w, tile_h, row, col)

            # ── 2. Optional resize (after crop) ─────────────────────
            if self.output_size is not None:
                tile = tile.resize(
                    (self.output_size, self.output_size),
                    RESAMPLE_FILTER,
                )

            # ── 3. Save ──────────────────────────────────────────────
            filename = f"{self.material_name}.{udim}.png"
            out_path = self.output_folder / filename
            tile.save(str(out_path), format="PNG")

            size_str = f"{tile.size[0]}×{tile.size[1]}"
            print(f"    [{row},{col}] {face_name:8s}  UDIM {udim}  {size_str:>12}  → {filename}")
            written.append(str(out_path))

        return written

    # ------------------------------------------------------------------
    def run(self) -> List[str]:
        """Alias for :meth:`split` with header/footer logging."""
        print("=" * 62)
        print("  Collage Splitter")
        print(f"  Collage       : {self.collage_path}")
        print(f"  Material name : {self.material_name}")
        print(f"  Face order    : {self.face_order}")
        print(f"  UDIM start    : {self.udim_start}")
        print(f"  Grid          : {self.cols} cols × {self.rows} rows")
        print("=" * 62)

        written = self.split()

        print()
        print(f"  Done — {len(written)} tiles written to:")
        print(f"  {self.output_folder.resolve()}")
        print("=" * 62)
        return written


# ---------------------------------------------------------------------------
# ── Batch helper ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def split_collage_set(
    collages:      dict,
    output_root:   str           = OUTPUT_ROOT,
    output_size:   Optional[int] = OUTPUT_SIZE,
    face_order:    List[str]     = FACE_ORDER,
    udim_start:    int           = UDIM_START,
) -> dict:
    """
    Convenience function to split multiple collages in one call.

    Parameters
    ----------
    collages : dict
        Mapping of ``material_name → collage_path``.  Example::

            {
                "robot_rgba"   : "./output/collage_rgba.png",
                "robot_depth"  : "./output/collage_depth.png",
                "robot_normals": "./output/collage_normals.png",
            }

    Returns
    -------
    dict mapping material_name → list of written file paths.
    """
    results = {}
    for material_name, collage_path in collages.items():
        splitter = CollageSplitter(
            collage_path  = collage_path,
            material_name = material_name,
            output_root   = output_root,
            output_size   = output_size,
            face_order    = face_order,
            udim_start    = udim_start,
        )
        results[material_name] = splitter.run()
    return results


# ---------------------------------------------------------------------------
# ── CLI entry-point ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="collage_splitter.py",
        description=(
            "Split a 3×2 PNG collage into 6 individual UDIM-named face images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "collage",
        metavar="COLLAGE_PNG",
        help="Path to the source 3×2 collage PNG.",
    )
    p.add_argument(
        "material_name",
        metavar="MATERIAL_NAME",
        help=(
            "Base name for the output files and sub-folder.  "
            "Files will be saved as {material_name}.{UDIM}.png."
        ),
    )
    p.add_argument(
        "--output-root", "-o",
        default=OUTPUT_ROOT,
        metavar="DIR",
        help="Root output directory.  A sub-folder is created inside it.",
    )
    p.add_argument(
        "--output-size", "-s",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Resize each tile to PX × PX after cropping.  "
            "Omit to keep the native tile size."
        ),
    )
    p.add_argument(
        "--face-order", "-f",
        nargs=6,
        default=FACE_ORDER,
        metavar="FACE",
        help=(
            "Six face names in collage order (left→right, top→bottom).  "
            f"Default: {' '.join(FACE_ORDER)}"
        ),
    )
    p.add_argument(
        "--udim-start", "-u",
        type=int,
        default=UDIM_START,
        metavar="N",
        help="First UDIM index (default 1001).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=3,
        metavar="N",
        help="Number of columns in the collage grid.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=2,
        metavar="N",
    )
    return p


if __name__ == "__main__":
    FACE_ORDER = ["right", "left", "top", "bottom", "front", "back"]
    splitter = CollageSplitter(
        collage_path  = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\output\collage_normals.png",
        material_name = "robot_normals",
        output_root   = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\output",
        output_size   = 2048,
        face_order    = FACE_ORDER,
        udim_start    = UDIM_START,
        cols          = 3,
        rows          = 2,
    )

    written = splitter.run()

    print("\nWritten files:")
    for path in written:
        print(f"  {path}")