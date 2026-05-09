
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# ── Global configuration ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

FACE_ORDER: List[str] = ["face_0", "face_1", "face_2", "face_3"]
OUTPUT_ROOT = "./output"
UDIM_START = 1001
OUTPUT_SIZE = None
RESAMPLE_FILTER = PILImage.LANCZOS

# ---------------------------------------------------------------------------
# ── Core helpers ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _tile_size(collage: PILImage.Image, cols: int = 2, rows: int = 2):
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

def _crop_tile(collage: PILImage.Image, tile_w: int, tile_h: int, row: int, col: int):
    left   = col * tile_w
    upper  = row * tile_h
    right  = left + tile_w
    lower  = upper + tile_h
    return collage.crop((left, upper, right, lower))

def _udim(index: int, start: int = UDIM_START):
    return start + index

# ---------------------------------------------------------------------------
# ── Main class ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class CollageSplitter:
    def __init__(
        self,
        collage_path,
        material_name,
        output_root = OUTPUT_ROOT,
        output_size = OUTPUT_SIZE,
        face_order = FACE_ORDER,
        udim_start = UDIM_START,
        cols = 2,
        rows = 2,
    ):
        if not os.path.isfile(collage_path):
            raise FileNotFoundError(f"Collage not found: {collage_path}")

        expected = cols * rows
        if len(face_order) != expected:
            raise ValueError(
                f"face_order must have exactly {expected} entries for a "
                f"{cols}×{rows} grid; got {len(face_order)}."
            )

        self.collage_path = collage_path
        self.material_name = material_name
        self.output_root = output_root
        self.output_size = output_size
        self.face_order = face_order
        self.udim_start = udim_start
        self.cols = cols
        self.rows = rows

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
    collages,
    output_root = OUTPUT_ROOT,
    output_size = OUTPUT_SIZE,
    face_order = FACE_ORDER,
    udim_start = UDIM_START,):

    results = {}
    for material_name, collage_path in collages.items():
        splitter = CollageSplitter(
            collage_path = collage_path,
            material_name = material_name,
            output_root = output_root,
            output_size = output_size,
            face_order = face_order,
            udim_start = udim_start,
        )
        results[material_name] = splitter.run()
    return results