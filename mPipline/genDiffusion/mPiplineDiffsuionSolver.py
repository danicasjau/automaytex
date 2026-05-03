import sys
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

import os
import gc
import math
import numpy as np
import torch
from PIL import Image, ImageFilter

from diffusers import StableDiffusionInpaintPipeline

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL_CACHE   = r"M:\ai\.models\checkpoints"

# SD 1.5 inpainting — ~4 GB VRAM in fp16, runs fine on RTX 4060 8 GB
# No ControlNets, no LoRA, no aux models.
INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Generation
DEFAULT_STEPS       = 25          # 20–30 is the sweet spot
DEFAULT_STRENGTH    = 0.7        # How aggressively seams are repainted (0.4–0.7)
DEFAULT_GUIDANCE    = 7.5         # CFG scale
TARGET_SIZE         = 1024        # Output image is always this square
TILE_SIZE           = 512         # SD 1.5 native tile size
TILE_OVERLAP        = 128         # Overlap between tiles to avoid hard seams
SEAM_FEATHER        = 48          # Gaussian blur radius for mask feathering


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _load_image(path: str, width: int = None, height: int = None) -> Image.Image:
    """Load and optionally resize image."""
    img = Image.open(path).convert("RGB")
    if width and height:
        return img.resize((width, height), Image.LANCZOS)
    return img


def _build_seam_mask(img_w: int, img_h: int,
                     tile_x: int, tile_y: int,
                     tile_w: int, tile_h: int,
                     feather: int = SEAM_FEATHER) -> Image.Image:
    """
    Build a feathered inpaint mask covering tile seam regions.
    The mask is white (255) where the model should repaint,
    black (0) where the original pixels are kept.
    Only the overlapping border strips are marked white.
    """
    mask = Image.new("L", (tile_w, tile_h), 0)
    arr  = np.zeros((tile_h, tile_w), dtype=np.uint8)

    # Mark overlap bands (top / left edges that come from a previous tile)
    if tile_x > 0:
        arr[:, :TILE_OVERLAP] = 255        # left seam band
    if tile_y > 0:
        arr[:TILE_OVERLAP, :] = 255        # top seam band

    mask = Image.fromarray(arr, mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask.convert("RGB")


def _alpha_blend_tile(canvas: np.ndarray,
                      tile:   np.ndarray,
                      tx: int, ty: int,
                      tile_w: int, tile_h: int,
                      overlap: int) -> np.ndarray:
    """
    Paste `tile` onto `canvas` at (tx, ty) with a linear cross-fade
    in the overlap region so seams are invisible.
    """
    region = canvas[ty:ty + tile_h, tx:tx + tile_w].copy()

    # Build a float weight mask: 0 → keep canvas, 1 → use tile
    weight = np.ones((tile_h, tile_w), dtype=np.float32)
    if tx > 0 and overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        weight[:, :overlap] *= ramp[np.newaxis, :]
    if ty > 0 and overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        weight[:overlap, :] *= ramp[:, np.newaxis]

    weight = weight[:, :, np.newaxis]   # broadcast over RGB channels
    blended = (tile * weight + region * (1.0 - weight)).astype(np.uint8)
    canvas[ty:ty + tile_h, tx:tx + tile_w] = blended
    return canvas


# ─── MAIN CLASS ───────────────────────────────────────────────────────────────

class TextureSeamFixer:
    """
    Uses SD 1.5 Inpainting to eliminate seams / inconsistencies
    in a 1024×1024 texture map.

    Works by splitting the image into overlapping 512×512 tiles,
    running diffusion inpainting on each seam region, and blending
    everything back together. No ControlNets, no LoRA.

    Fits in ~5–6 GB VRAM → comfortable on RTX 4060 8 GB.
    """

    def __init__(
        self,
        model_id:   str = INPAINT_MODEL_ID,
        cache_dir:  str = MODEL_CACHE,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[Init] device={self.device}  dtype={self.dtype}")

        print("[Init] Loading SD 1.5 Inpaint pipeline…")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
            safety_checker=None,       # skip NSFW filter — saves ~0.5 GB VRAM
            requires_safety_checker=False,
        ).to(self.device)

        # ── Memory optimisations for 8 GB GPUs ────────────────────────────────
        self.pipe.enable_attention_slicing(slice_size="auto")
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        try:
            import xformers  # noqa
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[Init] xformers enabled.")
        except ImportError:
            print("[Init] xformers not available — using PyTorch SDPA.")
            # torch >= 2.0 has built-in memory-efficient attention
            try:
                self.pipe.unet.set_attn_processor(
                    torch.nn.attention.SDPABackend  # type: ignore
                )
            except Exception:
                pass

        print("[Init] Pipeline ready.")

    # ── Tile-based inpainting ──────────────────────────────────────────────────

    @torch.no_grad()
    def _inpaint_tile(
        self,
        image: Image.Image,
        mask:  Image.Image,
        prompt: str,
        neg_prompt: str,
        steps: int,
        strength: float,
        guidance: float,
    ) -> Image.Image:
        """Run inpainting on a single 512×512 tile."""
        result = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            mask_image=mask,
            height=TILE_SIZE,
            width=TILE_SIZE,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=guidance,
        ).images[0]
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def fix_texture(
        self,
        input_path:  str,
        output_path: str,
        prompt:      str,
        neg_prompt:  str = (
            "seam, tiling artifact, edge, border, blur, glitch, "
            "watermark, text, oversaturated, cartoon, low quality"
        ),
        steps:       int   = DEFAULT_STEPS,
        strength:    float = DEFAULT_STRENGTH,
        guidance:    float = DEFAULT_GUIDANCE,
        target_width: int  = None,
        target_height: int = None,
    ) -> Image.Image:
        """
        Load a texture, fix seams with diffusion inpainting, save result.

        Args:
            input_path    Path to the input texture image.
            output_path   Where to write the fixed texture.
            prompt        Texture description.
            neg_prompt    What to avoid.
            steps         Denoising steps per tile.
            strength      How much to repaint.
            guidance      CFG guidance scale.
            target_width  Optional resize width.
            target_height Optional resize height.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        print(f"[Fix] Loading {input_path}")
        base_img = _load_image(input_path, target_width, target_height)
        img_w, img_h = base_img.size
        canvas   = np.array(base_img, dtype=np.uint8)

        # ── Tile grid ──────────────────────────────────────────────────────────
        # We tile with a stride < tile size so seams overlap previous tiles.
        stride   = TILE_SIZE - TILE_OVERLAP
        n_cols   = math.ceil((img_w - TILE_OVERLAP) / stride)
        n_rows   = math.ceil((img_h - TILE_OVERLAP) / stride)
        total    = n_cols * n_rows
        print(f"[Fix] Grid {n_cols}×{n_rows} ({total} tiles), "
              f"tile={TILE_SIZE}, overlap={TILE_OVERLAP}, stride={stride}")

        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col
                tx  = min(col * stride, img_w - TILE_SIZE)
                ty  = min(row * stride, img_h - TILE_SIZE)

                print(f"[Tile {idx+1}/{total}] x={tx} y={ty}")

                # Crop tile from current canvas state
                tile_pil = Image.fromarray(
                    canvas[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                ).convert("RGB")

                # Build seam mask (only paint over overlap bands, not whole tile)
                mask_pil = _build_seam_mask(
                    img_w, img_h, tx, ty,
                    TILE_SIZE, TILE_SIZE,
                    feather=SEAM_FEATHER,
                )

                # Skip tiles that have no seam to fix (top-left first tile)
                mask_arr = np.array(mask_pil.convert("L"))
                if mask_arr.max() == 0:
                    print(f"  → No seam, skipping.")
                    continue

                # Inpaint
                fixed_tile = self._inpaint_tile(
                    tile_pil, mask_pil,
                    prompt, neg_prompt,
                    steps, strength, guidance,
                )

                # Blend back with feathered alpha
                fixed_arr = np.array(fixed_tile.resize(
                    (TILE_SIZE, TILE_SIZE), Image.LANCZOS
                ), dtype=np.uint8)

                canvas = _alpha_blend_tile(
                    canvas, fixed_arr, tx, ty,
                    TILE_SIZE, TILE_SIZE, TILE_OVERLAP,
                )

                # Free VRAM between tiles
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        result = Image.fromarray(canvas)
        result.save(output_path)
        print(f"[Done] Saved → {output_path}")
        return result


# ─── EXECUTION ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix texture seams using SD 1.5 diffusion inpainting."
    )
    parser.add_argument("--input",    required=True,  help="Path to input texture")
    parser.add_argument("--output",   required=True,  help="Path for fixed texture")
    parser.add_argument("--prompt",   required=True,  help="Texture description")
    parser.add_argument(
        "--neg_prompt", default=(
            "seam, tiling artifact, edge, border, blur, glitch, "
            "watermark, text, oversaturated, cartoon, low quality"
        ),
        help="Negative prompt",
    )
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--width",    type=int,   default=None)
    parser.add_argument("--height",   type=int,   default=None)
    args = parser.parse_args()

    fixer = TextureSeamFixer()
    fixer.fix_texture(
        input_path  = args.input,
        output_path = args.output,
        prompt      = args.prompt,
        neg_prompt  = args.neg_prompt,
        steps       = args.steps,
        strength    = args.strength,
        guidance    = args.guidance,
        target_width  = args.width,
        target_height = args.height,
    )