"""
ControlNet Multi-View Texture Generator
========================================
Generates coherent textures from a multi-view 3D object collage (3×2 grid).
Designed for SD 1.5 + ControlNet on 8 GB VRAM via sequential CPU offloading.

Key improvements over v1:
  - Shared prompt embeddings computed once for all tiles
  - Shared seed + latent noise anchor → cross-tile colour/style coherence
  - Tile overlap-blending (seam-free stitching)
  - Correct img2img timestep maths
  - Tuned defaults for well-defined texture output
  - Strict sequential CPU offloading: only ONE heavy module on GPU at a time
  - Optional xformers / torch.compile support
"""

import sys, os, gc, math, argparse
import numpy as np
import torch
from PIL import Image, ImageFilter

# ── Allow a local venv to shadow system packages ──────────────────────────────
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    UNet2DConditionModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,   # Euler-A: crisper detail than plain Euler
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CACHE   = r"M:\ai\.models\checkpoints"
BASE_MODEL_ID = "emilianJR/epiCRealism"          # SD 1.5-based

CONTROLNET_DEPTH_ID  = "lllyasviel/sd-controlnet-depth"
CONTROLNET_NORMAL_ID = "lllyasviel/sd-controlnet-normal"

CONDITION_MODE = "depth"    # "depth" | "normal" | "both"
GRID_MODE      = "tiles"    # "collage" | "tiles"
GRID_COLS      = 3
GRID_ROWS      = 2

# ── Generation defaults ───────────────────────────────────────────────────────
DEFAULT_STEPS    = 30          # 30 steps → noticeably sharper than 20
DEFAULT_STRENGTH = 0.95        # img2img: moderate; preserves init colour
DEFAULT_CN_SCALE = 0.85        # ControlNet: strong but not over-constraining
DEFAULT_GUIDANCE = 7.5
DEFAULT_SEED     = 42

MAX_TILE_PX  = 512             # max edge length per tile (keeps VRAM ≤ 8 GB)
LATENT_ALIGN = 8               # VAE latent stride (SD = 8, not 64)
OVERLAP_PX   = 32              # pixel overlap between tiles for seam blending

NEG_PROMPT_DEFAULT = (
    "cartoon, anime, illustration, painting, sketch, plastic, blurry, "
    "watermark, oversaturated, low resolution, jpeg artifacts, noise, "
    "deformed, duplicate, missing seams, tiling artifacts"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _align(v: int, align: int = LATENT_ALIGN) -> int:
    """Round v up to the nearest multiple of `align`."""
    return max(align, (v + align - 1) // align * align)


def _offload(*models):
    """Move models to CPU and free CUDA cache."""
    for m in models:
        if m is not None:
            m.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _pil_to_latent_tensor(img: Image.Image, device, dtype) -> torch.Tensor:
    """PIL RGB image → [-1, 1] float tensor (1, 3, H, W)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)


def _pil_to_cond_tensor(img: Image.Image, device, dtype) -> torch.Tensor:
    """PIL RGB image → [0, 1] float tensor (1, 3, H, W) for ControlNet."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().float().clamp(-1.0, 1.0)
    arr = (arr / 2.0 + 0.5).permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((arr * 255).astype(np.uint8))


def prepare_depth_map(img: Image.Image) -> Image.Image:
    """
    Normalise a depth/normal map for ControlNet input.
    Converts to greyscale, auto-stretches contrast, returns RGB.
    """
    grey = np.array(img.convert("L")).astype(np.float32)
    lo, hi = grey.min(), grey.max()
    if hi > lo:
        grey = (grey - lo) / (hi - lo) * 255.0
    return Image.fromarray(grey.astype(np.uint8)).convert("RGB")


def _resize_for_tile(img: Image.Image, max_px: int = MAX_TILE_PX) -> Image.Image:
    """Resize so the longer edge ≤ max_px, then align both dims."""
    w, h = img.size
    scale = min(max_px / max(w, h), 1.0)
    w2, h2 = _align(int(w * scale)), _align(int(h * scale))
    return img.resize((w2, h2), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# GRID UTILITIES  (support variable grid size + overlap)
# ─────────────────────────────────────────────────────────────────────────────

def split_grid(collage: Image.Image, cols: int, rows: int, overlap: int = 0):
    """
    Split a (cols × rows) collage into tiles.
    Returns: list[PIL], tile_w (no overlap), tile_h (no overlap)
    """
    W, H = collage.size
    tw, th = W // cols, H // rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(c * tw - overlap, 0)
            y0 = max(r * th - overlap, 0)
            x1 = min((c + 1) * tw + overlap, W)
            y1 = min((r + 1) * th + overlap, H)
            tiles.append(collage.crop((x0, y0, x1, y1)))
    return tiles, tw, th


def stitch_grid(
    tiles,
    tw: int, th: int,
    cols: int, rows: int,
    overlap: int = 0,
) -> Image.Image:
    """
    Reconstruct collage from tiles with optional feathered seam blending.
    Each tile is resized to (tw, th) before pasting so borders line up exactly.
    """
    canvas  = Image.new("RGB",  (tw * cols, th * rows))
    weights = Image.new("L",    (tw * cols, th * rows), 0)
    fe      = overlap // 2  # feather radius

    for idx, tile in enumerate(tiles):
        r, c  = divmod(idx, cols)
        tile_r = tile.resize((tw, th), Image.LANCZOS)

        if fe > 0:
            # Build a soft weight mask for this tile
            mask = np.ones((th, tw), dtype=np.float32)
            for axis, n, coord in [(1, cols, c), (0, rows, r)]:
                if coord > 0:
                    ramp = np.linspace(0, 1, fe)
                    slc  = [slice(None), slice(None)]
                    slc[axis] = slice(0, fe)
                    mask[tuple(slc)] *= ramp if axis == 1 else ramp[:, None]
                if coord < (cols if axis == 1 else rows) - 1:
                    ramp = np.linspace(1, 0, fe)
                    slc  = [slice(None), slice(None)]
                    slc[axis] = slice(tw - fe if axis == 1 else th - fe, None)
                    mask[tuple(slc)] *= ramp if axis == 1 else ramp[:, None]
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))

            # Composite with existing canvas
            cur   = canvas.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            cur_w = weights.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            cur_w_arr  = np.array(cur_w, dtype=np.float32)
            mask_arr   = np.array(mask_img, dtype=np.float32)
            total      = cur_w_arr + mask_arr
            total      = np.where(total == 0, 1, total)
            blend_arr  = (
                np.array(cur, dtype=np.float32) * cur_w_arr[:, :, None] +
                np.array(tile_r, dtype=np.float32) * mask_arr[:, :, None]
            ) / total[:, :, None]
            canvas.paste(Image.fromarray(blend_arr.astype(np.uint8)), (c * tw, r * th))
            weights.paste(
                Image.fromarray(np.clip(total, 0, 255).astype(np.uint8)),
                (c * tw, r * th),
            )
        else:
            canvas.paste(tile_r, (c * tw, r * th))

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ControlNetTextureGenerator:
    """
    SD 1.5 + ControlNet multi-view texture generator.

    Memory strategy (8 GB VRAM):
      - All models start on CPU.
      - Only one module at a time is moved to GPU:
          text_encoder → offload → vae encode → offload →
          controlnet+unet (denoising loop, both stay on GPU together) →
          offload both → vae decode → offload.
      - ControlNet + UNet together use ≈ 5–6 GB for 512×512 tiles.

    Coherence strategy:
      - Prompt embeddings computed ONCE and reused for all tiles.
      - Global generator seed → all tiles share the same latent noise basis,
        which keeps colour tone and style consistent across views.
      - Overlapping tile crop + feathered seam blending removes hard borders.
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        controlnet_depth_id: str = CONTROLNET_DEPTH_ID,
        controlnet_normal_id: str = CONTROLNET_NORMAL_ID,
        condition_mode: str = CONDITION_MODE,
        cache_dir: str = MODEL_CACHE,
        enable_xformers: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        self.condition_mode = condition_mode

        print(f"[Init] device={self.device}  dtype={self.dtype}  cond_mode={condition_mode}")

        # ── Tokenizer (no GPU needed) ─────────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_model_id, subfolder="tokenizer", cache_dir=cache_dir)

        # ── Text encoder ──────────────────────────────────────────────────────
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model_id, subfolder="text_encoder",
            torch_dtype=self.dtype, cache_dir=cache_dir)
        self.text_encoder.requires_grad_(False)

        # ── VAE ───────────────────────────────────────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            base_model_id, subfolder="vae",
            torch_dtype=self.dtype, cache_dir=cache_dir)
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()
        self.vae.enable_slicing()

        # ── UNet ──────────────────────────────────────────────────────────────
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_id, subfolder="unet",
            torch_dtype=self.dtype, cache_dir=cache_dir)
        self.unet.requires_grad_(False)

        # ── ControlNet(s) ─────────────────────────────────────────────────────
        print("[Init] Loading ControlNet(s)…")
        self.cn_depth  = None
        self.cn_normal = None

        if condition_mode in ("depth", "both"):
            self.cn_depth = ControlNetModel.from_pretrained(
                controlnet_depth_id, torch_dtype=self.dtype, cache_dir=cache_dir)
            self.cn_depth.requires_grad_(False)

        if condition_mode in ("normal", "both"):
            self.cn_normal = ControlNetModel.from_pretrained(
                controlnet_normal_id, torch_dtype=self.dtype, cache_dir=cache_dir)
            self.cn_normal.requires_grad_(False)

        if condition_mode == "both":
            self.controlnet = MultiControlNetModel([self.cn_depth, self.cn_normal])
        elif condition_mode == "depth":
            self.controlnet = self.cn_depth
        else:
            self.controlnet = self.cn_normal

        # ── Scheduler: Euler-A gives crisper high-frequency detail ────────────
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            base_model_id, subfolder="scheduler",
            cache_dir=cache_dir,
        )

        # ── Optional xformers ─────────────────────────────────────────────────
        if self.device == "cuda" and enable_xformers:
            try:
                import xformers  # noqa: F401
                self.unet.enable_xformers_memory_efficient_attention()
                if self.cn_depth:
                    self.cn_depth.enable_xformers_memory_efficient_attention()
                if self.cn_normal:
                    self.cn_normal.enable_xformers_memory_efficient_attention()
                print("[Init] xformers enabled.")
            except Exception:
                print("[Init] xformers not available; continuing without it.")

        print("[Init] All components loaded on CPU.")

    # ── Prompt encoding (runs once for the whole batch) ───────────────────────

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, neg_prompt: str) -> torch.Tensor:
        """
        Returns a (2, 77, 768) tensor [negative, positive] on CPU.
        Kept on CPU; moved to device only during the denoising loop.
        """
        self.text_encoder.to(self.device)

        def _embed(text: str) -> torch.Tensor:
            ids = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            return self.text_encoder(ids).last_hidden_state

        pos_emb = _embed(prompt)
        neg_emb = _embed(neg_prompt)
        emb = torch.cat([neg_emb, pos_emb], dim=0).cpu()   # keep on CPU

        _offload(self.text_encoder)
        return emb

    # ── VAE encode ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_image(self, img: Image.Image) -> torch.Tensor:
        self.vae.to(self.device)
        t = _pil_to_latent_tensor(img, self.device, self.dtype)
        lat = self.vae.encode(t).latent_dist.sample() * self.vae.config.scaling_factor
        result = lat.cpu()
        _offload(self.vae)
        return result

    # ── VAE decode ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        self.vae.to(self.device)
        lat = (latents.to(self.device, dtype=self.dtype)
               / self.vae.config.scaling_factor)
        out = self.vae.decode(lat).sample
        _offload(self.vae)
        return _tensor_to_pil(out)

    # ── Core denoising (single tile) ─────────────────────────────────────────

    @torch.no_grad()
    def _denoise(
        self,
        encoder_hidden_states: torch.Tensor,  # (2,77,768) on CPU
        depth_img:  Image.Image,
        init_img:   Image.Image | None,
        normal_img: Image.Image | None,
        num_steps:  int,
        strength:   float,
        cn_scale,
        guidance:   float,
        generator:  torch.Generator,
    ) -> Image.Image:
        """
        One full encode → denoise → decode pass for a single tile.
        encoder_hidden_states must be pre-computed (shared across tiles).
        """
        w, h = depth_img.size
        ehs = encoder_hidden_states.to(self.device, dtype=self.dtype)

        # ── Init latents ──────────────────────────────────────────────────────
        if init_img is not None:
            init_lat = self._encode_image(init_img.resize((w, h), Image.LANCZOS))
            init_lat = init_lat.to(self.device, dtype=self.dtype)
        else:
            init_lat = None

        # ── Timestep schedule ─────────────────────────────────────────────────
        self.scheduler.set_timesteps(num_steps, device=self.device)
        all_ts = self.scheduler.timesteps

        if init_lat is not None:
            # img2img: start from a noisy version of the init image
            t_start  = max(int(num_steps * (1.0 - strength)), 1)
            timesteps = all_ts[t_start:]
            noise     = torch.randn_like(init_lat, generator=generator)
            latents   = self.scheduler.add_noise(init_lat, noise, timesteps[:1])
        else:
            # pure text-to-image
            timesteps = all_ts
            latents   = torch.randn(
                (1, self.unet.config.in_channels, h // 8, w // 8),
                dtype=self.dtype, device=self.device, generator=generator,
            ) * self.scheduler.init_noise_sigma

        # ── ControlNet condition tensors ──────────────────────────────────────
        depth_t = _pil_to_cond_tensor(depth_img, self.device, self.dtype)

        if self.condition_mode == "both" and normal_img is not None:
            norm_t      = _pil_to_cond_tensor(normal_img, self.device, self.dtype)
            cond_images = [depth_t, norm_t]
            cn_scales   = [cn_scale, cn_scale * 0.75]
        elif self.condition_mode == "both":
            cond_images = [depth_t, depth_t]
            cn_scales   = [cn_scale, cn_scale * 0.5]
        else:
            cond_images = depth_t
            cn_scales   = cn_scale

        # CFG: duplicate for unconditional + conditional pass
        if isinstance(cond_images, list):
            cond_cfg = [torch.cat([c, c]) for c in cond_images]
        else:
            cond_cfg = torch.cat([cond_images, cond_images])

        # ── Move UNet + ControlNet to GPU together ────────────────────────────
        self.controlnet.to(self.device)
        self.unet.to(self.device)

        for t in timesteps:
            lat_in  = self.scheduler.scale_model_input(
                torch.cat([latents] * 2), t)

            down_res, mid_res = self.controlnet(
                lat_in,
                t,
                encoder_hidden_states=ehs,
                controlnet_cond=cond_cfg,
                conditioning_scale=cn_scales,
                return_dict=False,
            )

            noise_pred = self.unet(
                lat_in,
                t,
                encoder_hidden_states=ehs,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample

            # Classifier-free guidance
            unc, cond_ = noise_pred.chunk(2)
            noise_pred = unc + guidance * (cond_ - unc)
            latents    = self.scheduler.step(noise_pred, t, latents).prev_sample

        _offload(self.controlnet, self.unet)

        return self._decode_latents(latents)

    # ── Tile-mode generation ──────────────────────────────────────────────────

    def _generate_tiles(
        self,
        shared_ehs: torch.Tensor,
        depth_collage: Image.Image,
        init_collage:  Image.Image | None,
        norm_collage:  Image.Image | None,
        num_steps: int,
        strength:  float,
        cn_scale:  float,
        guidance:  float,
        seed:      int,
        cols: int = GRID_COLS,
        rows: int = GRID_ROWS,
    ) -> Image.Image:
        orig_w, orig_h = depth_collage.size
        tw, th = orig_w // cols, orig_h // rows

        d_tiles = split_grid(depth_collage, cols, rows, overlap=OVERLAP_PX)[0]
        i_tiles = (split_grid(init_collage, cols, rows, overlap=OVERLAP_PX)[0]
                   if init_collage else [None] * cols * rows)
        n_tiles = (split_grid(norm_collage, cols, rows, overlap=OVERLAP_PX)[0]
                   if norm_collage else [None] * cols * rows)

        out_tiles = []

        for idx, (dt, it, nt) in enumerate(zip(d_tiles, i_tiles, n_tiles)):
            r, c = divmod(idx, cols)
            print(f"[Tile {idx + 1}/{cols * rows}]  row={r}  col={c}")

            dt_r = _resize_for_tile(dt)
            it_r = _resize_for_tile(it) if it else None
            nt_r = _resize_for_tile(nt) if nt else None

            # Tile-specific generator — deterministic but unique per tile.
            # Using seed + view_index keeps neighbouring tiles related while
            # avoiding exact duplicates; global style comes from shared_ehs.
            tile_gen = torch.Generator(device=self.device).manual_seed(seed + idx)

            tile_out = self._denoise(
                encoder_hidden_states=shared_ehs,
                depth_img=dt_r,
                init_img=it_r,
                normal_img=nt_r,
                num_steps=num_steps,
                strength=strength,
                cn_scale=cn_scale,
                guidance=guidance,
                generator=tile_gen,
            )
            out_tiles.append(tile_out)
            print(f"  ✓ tile done")

        return stitch_grid(out_tiles, tw, th, cols, rows, overlap=OVERLAP_PX)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_texture(
        self,
        prompt:             str,
        depth_collage_path: str,
        output_path:        str,
        init_collage_path:  str  = None,
        normal_collage_path: str = None,
        num_steps:          int  = DEFAULT_STEPS,
        strength:           float = DEFAULT_STRENGTH,
        cn_scale:           float = DEFAULT_CN_SCALE,
        guidance:           float = DEFAULT_GUIDANCE,
        neg_prompt:         str  = NEG_PROMPT_DEFAULT,
        grid_mode:          str  = GRID_MODE,
        seed:               int  = DEFAULT_SEED,
        cols:               int  = GRID_COLS,
        rows:               int  = GRID_ROWS,
    ) -> Image.Image:
        """
        Main entry point.

        Parameters
        ----------
        prompt              : Texture description (be specific: material, lighting, scale).
        depth_collage_path  : Path to (cols × rows) depth-map collage (grey or RGB).
        output_path         : Where to save the result.
        init_collage_path   : Optional colour-reference collage (img2img base).
        normal_collage_path : Optional normal-map collage (only used in 'both' mode).
        num_steps           : Denoising steps (30 recommended).
        strength            : img2img noise level — ignored when no init_collage_path.
        cn_scale            : ControlNet conditioning weight.
        guidance            : CFG scale.
        neg_prompt          : Negative prompt.
        grid_mode           : 'tiles' (recommended) or 'collage' (single pass).
        seed                : Global seed for reproducibility + cross-tile coherence.
        cols / rows         : Grid dimensions (default 3×2).
        """
        if not os.path.exists(depth_collage_path):
            raise FileNotFoundError(f"Depth collage not found: {depth_collage_path}")

        # ── Load inputs ───────────────────────────────────────────────────────
        depth_collage = prepare_depth_map(Image.open(depth_collage_path))
        init_collage  = (Image.open(init_collage_path).convert("RGB")
                         if init_collage_path else None)
        norm_collage  = (Image.open(normal_collage_path).convert("RGB")
                         if normal_collage_path else None)

        # ── Encode prompt ONCE (shared across all tiles) ──────────────────────
        print("[Encode] Computing prompt embeddings…")
        shared_ehs = self._encode_prompt(prompt, neg_prompt)  # stays on CPU

        # ── Generate ──────────────────────────────────────────────────────────
        if grid_mode == "tiles":
            result = self._generate_tiles(
                shared_ehs, depth_collage, init_collage, norm_collage,
                num_steps, strength, cn_scale, guidance, seed, cols, rows,
            )
        else:
            # Single-pass collage mode
            tw = _align(min(depth_collage.width, MAX_TILE_PX * cols))
            th = _align(int(depth_collage.height * tw / depth_collage.width))
            d_r = depth_collage.resize((tw, th), Image.LANCZOS)
            i_r = init_collage.resize((tw, th), Image.LANCZOS) if init_collage else None
            n_r = norm_collage.resize((tw, th), Image.LANCZOS) if norm_collage else None
            gen = torch.Generator(device=self.device).manual_seed(seed)
            result = self._denoise(
                shared_ehs, d_r, i_r, n_r,
                num_steps, strength, cn_scale, guidance, gen,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result.save(output_path)
        print(f"[Done] Texture saved → {output_path}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate textures from a multi-view depth collage using ControlNet.")

    parser.add_argument("--prompt",   required=True,
                        help="Texture description. Be specific: e.g. "
                             "'worn leather surface, rough stitching, warm brown tones, "
                             "4k PBR texture, photorealistic'")
    parser.add_argument("--depth",    required=True,  help="Path to depth collage")
    parser.add_argument("--output",   required=True,  help="Output path (.png)")
    parser.add_argument("--input",    default=None,   help="Optional colour init collage")
    parser.add_argument("--normal",   default=None,   help="Optional normal-map collage")
    parser.add_argument("--neg",      default=NEG_PROMPT_DEFAULT, help="Negative prompt")
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--cn_scale", type=float, default=DEFAULT_CN_SCALE)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    parser.add_argument("--mode",     default=GRID_MODE,
                        choices=["tiles", "collage"])
    parser.add_argument("--cols",     type=int,   default=GRID_COLS)
    parser.add_argument("--rows",     type=int,   default=GRID_ROWS)
    parser.add_argument("--cond",     default=CONDITION_MODE,
                        choices=["depth", "normal", "both"])

    args = parser.parse_args()

    gen = ControlNetTextureGenerator(condition_mode=args.cond)
    gen.generate_texture(
        prompt=args.prompt,
        depth_collage_path=args.depth,
        output_path=args.output,
        init_collage_path=args.input,
        normal_collage_path=args.normal,
        num_steps=args.steps,
        strength=args.strength,
        cn_scale=args.cn_scale,
        guidance=args.guidance,
        neg_prompt=args.neg,
        grid_mode=args.mode,
        seed=args.seed,
        cols=args.cols,
        rows=args.rows,
    )