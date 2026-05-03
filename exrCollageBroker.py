
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
FACE_ORDER: List[str] = ["face_0", "face_1", "face_2", "face_3"]

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

def _tile_size(collage: PILImage.Image, cols: int = 2, rows: int = 2) -> Tuple[int, int]:
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


def _crop_tile(collage: PILImage.Image, tile_w: int, tile_h: int, row: int, col: int) -> PILImage.Image:
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
    def __init__(
        self,
        collage_path:  str,
        material_name: str,
        output_root:   str            = OUTPUT_ROOT,
        output_size:   Optional[int]  = OUTPUT_SIZE,
        face_order:    List[str]      = FACE_ORDER,
        udim_start:    int            = UDIM_START,
        cols:          int            = 2,
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
                tile = tile.resize((self.output_size, self.output_size), RESAMPLE_FILTER)

            # ── 3. Save ──────────────────────────────────────────────
            filename = f"{self.material_name}.{udim}.png"
            out_path = self.output_folder / filename
            tile.save(str(out_path), format="PNG")

            import subprocess
            import os
            try:
                from config import configuration
                cfg = configuration()
                script_path = os.path.join(cfg.base_dir, "upscalerPolishing.py")
                cmd = [
                    cfg.python_exe, script_path,
                    "--input", str(out_path),
                    "--output", str(self.output_folder),
                    "--res", "4k"
                ]
                print(f"    [Upscaler] Running subprocess...")
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"    [ERROR] Upscaler failed: {e}")

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
            "Split a 2×2 PNG collage into 4 individual UDIM-named face images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "collage",
        metavar="COLLAGE_PNG",
        help="Path to the source 2×2 collage PNG.",
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
        nargs=4,
        default=FACE_ORDER,
        metavar="FACE",
        help=(
            "Four face names in collage order (left→right, top→bottom).  "
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
        default=2,
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
    FACE_ORDER = ["face_0", "face_1", "face_2", "face_3"]
    splitter = CollageSplitter(
        collage_path  = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\output\collage_normals.png",
        material_name = "robot_normals",
        output_root   = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\output",
        output_size   = 2048,
        face_order    = FACE_ORDER,
        udim_start    = UDIM_START,
        cols          = 2,
        rows          = 2,
    )

    written = splitter.run()

    print("\nWritten files:")
    for path in written:
        print(f"  {path}")