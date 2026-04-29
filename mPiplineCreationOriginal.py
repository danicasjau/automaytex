
import sys

sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

import os
import gc
import numpy as np
import torch

from PIL import Image

from transformers import CLIPTokenizer, CLIPTextModel

from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    UNet2DConditionModel,
    AutoencoderKL,
    EulerDiscreteScheduler,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL_CACHE   = r"M:\ai\.models\checkpoints"
BASE_MODEL_ID = "emilianJR/epiCRealism"          # SD1.5-based model

# ControlNet checkpoints (SD1.5 flavour)
CONTROLNET_DEPTH_ID  = "lllyasviel/sd-controlnet-depth"
CONTROLNET_NORMAL_ID = "lllyasviel/sd-controlnet-normal"

CONDITION_MODE = "depth"   # "depth" | "normal" | "both"
GRID_MODE      = "tiles"   # "collage" | "tiles"

# Generation
DEFAULT_STEPS        = 20          # More steps = better quality; 20 is a good balance
DEFAULT_STRENGTH     = 0.75        # img2img noise strength (0 = no change, 1 = full noise)
DEFAULT_CN_SCALE     = 0.8         # ControlNet conditioning scale (0–1)
DEFAULT_GUIDANCE     = 7.5         # CFG guidance scale
LATENT_ALIGN         = 64          # SD latent grid size; keep as is
MAX_TILE_WIDTH       = 512         # Max per-tile width for tile mode


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _to_device(model, device):
    """Move model to device, return model."""
    return model.to(device)

def _offload(model, dtype):
    """Move model back to CPU and empty CUDA cache."""
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _align(v: int, align: int = LATENT_ALIGN) -> int:
    return max(align, (v + align - 1) // align * align)

def _load_resize(path: str, max_w: int = MAX_TILE_WIDTH) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_w:
        h = int(h * max_w / w)
        w = max_w
    w, h = _align(w), _align(h)
    return img.resize((w, h), Image.LANCZOS)

def _pil_to_tensor(img: Image.Image, device, dtype) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().cpu().float().clamp(-1, 1) / 2 + 0.5)
    arr = arr.permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((arr * 255).astype(np.uint8))

def prepare_depth_map(depth_img: Image.Image) -> Image.Image:
    """
    Ensure depth map is in 0-255 greyscale, then convert to RGB
    (ControlNet-depth expects 3-channel input).
    If your depth is stored as EXR or 16-bit PNG, convert before calling.
    """
    grey = depth_img.convert("L")
    return grey.convert("RGB")

def split_grid(collage: Image.Image, cols: int = 3, rows: int = 2):
    """Split a 3×2 collage into (cols*rows) tiles."""
    w, h = collage.size
    tw, th = w // cols, h // rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
            tiles.append(collage.crop(box))
    return tiles, tw, th

def stitch_grid(tiles, tw: int, th: int, cols: int = 3, rows: int = 2) -> Image.Image:
    """Reconstruct a (cols×rows) collage from individual tiles."""
    canvas = Image.new("RGB", (tw * cols, th * rows))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        tile_resized = tile.resize((tw, th), Image.LANCZOS)
        canvas.paste(tile_resized, (c * tw, r * th))
    return canvas


# ─── MAIN CLASS ───────────────────────────────────────────────────────────────

class ControlNetTextureGenerator:
    """
    SD 1.5 + ControlNet texture generator.
    All heavy models live on CPU when idle; only the active component
    is pushed to GPU. This keeps peak VRAM below ~7 GB for tile mode.
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        controlnet_depth_id: str = CONTROLNET_DEPTH_ID,
        controlnet_normal_id: str = CONTROLNET_NORMAL_ID,
        condition_mode: str = CONDITION_MODE,
        cache_dir: str = MODEL_CACHE,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        self.condition_mode = condition_mode
        print(f"[Init] device={self.device}  dtype={self.dtype}  cond_mode={condition_mode}")

        # ── Text encoder ──────────────────────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_model_id, subfolder="tokenizer", cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model_id, subfolder="text_encoder",
            torch_dtype=self.dtype, cache_dir=cache_dir)

        # ── VAE ───────────────────────────────────────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            base_model_id, subfolder="vae",
            torch_dtype=self.dtype, cache_dir=cache_dir)
        self.vae.enable_tiling()
        self.vae.enable_slicing()

        # ── UNet ──────────────────────────────────────────────────────────────
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_id, subfolder="unet",
            torch_dtype=self.dtype, cache_dir=cache_dir)

        # ── ControlNet(s) ─────────────────────────────────────────────────────
        print("[Init] Loading ControlNet(s)…")
        if condition_mode in ("depth", "both"):
            self.cn_depth = ControlNetModel.from_pretrained(
                controlnet_depth_id, torch_dtype=self.dtype, cache_dir=cache_dir)
        else:
            self.cn_depth = None

        if condition_mode in ("normal", "both"):
            self.cn_normal = ControlNetModel.from_pretrained(
                controlnet_normal_id, torch_dtype=self.dtype, cache_dir=cache_dir)
        else:
            self.cn_normal = None

        # MultiControlNet wrapper used in "both" mode
        if condition_mode == "both":
            self.controlnet = MultiControlNetModel([self.cn_depth, self.cn_normal])
        elif condition_mode == "depth":
            self.controlnet = self.cn_depth
        else:
            self.controlnet = self.cn_normal

        # ── Scheduler ─────────────────────────────────────────────────────────
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_id, subfolder="scheduler",
            cache_dir=cache_dir, timestep_spacing="trailing")

        # ── Optional xformers ─────────────────────────────────────────────────
        if self.device == "cuda":
            try:
                import xformers
                self.unet.enable_xformers_memory_efficient_attention()
                if self.cn_depth:
                    self.cn_depth.enable_xformers_memory_efficient_attention()
                if self.cn_normal:
                    self.cn_normal.enable_xformers_memory_efficient_attention()
                print("[Init] xformers enabled.")
            except Exception:
                print("[Init] xformers not available.")

        print("[Init] All components loaded on CPU.")

    # ── Prompt encoding ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, neg_prompt: str):
        self.text_encoder.to(self.device)

        def _embed(p):
            toks = self.tokenizer(
                p, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(self.device)
            return self.text_encoder(toks.input_ids).last_hidden_state

        pos = _embed(prompt)
        neg = _embed(neg_prompt)

        _offload(self.text_encoder, self.dtype)
        return torch.cat([neg, pos], dim=0)   # (2, 77, 768)

    # ── VAE helpers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_image(self, img: Image.Image) -> torch.Tensor:
        self.vae.to(self.device)
        t = _pil_to_tensor(img, self.device, self.dtype)
        latents = self.vae.encode(t).latent_dist.sample() * self.vae.config.scaling_factor
        _offload(self.vae, self.dtype)
        return latents

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        self.vae.to(self.device)
        latents = latents / self.vae.config.scaling_factor
        out = self.vae.decode(latents).sample
        _offload(self.vae, self.dtype)
        return _tensor_to_pil(out)

    # ── ControlNet condition tensor ───────────────────────────────────────────

    def _cond_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a [1, 3, H, W] tensor normalised to [0, 1]."""
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device, dtype=self.dtype)

    # ── Single-image generation ───────────────────────────────────────────────

    @torch.no_grad()
    def _generate_single(
        self,
        prompt: str,
        neg_prompt: str,
        depth_img: Image.Image,          # pre-processed depth (RGB)
        init_img: Image.Image,           # colour reference (or None for t2i)
        normal_img: Image.Image = None,  # optional, used in "both" mode
        num_steps: int = DEFAULT_STEPS,
        strength: float = DEFAULT_STRENGTH,
        cn_scale=DEFAULT_CN_SCALE,
        guidance: float = DEFAULT_GUIDANCE,
    ) -> Image.Image:

        w, h = depth_img.size

        # 1. Text embeddings
        encoder_hidden_states = self._encode_prompt(prompt, neg_prompt)

        # 2. Init latents + noise
        if init_img is not None:
            init_latents = self._encode_image(init_img.resize((w, h), Image.LANCZOS))
        else:
            init_latents = torch.randn(
                (1, self.unet.config.in_channels, h // 8, w // 8),
                dtype=self.dtype, device=self.device)

        # 3. Timestep schedule (img2img subset when init_img given)
        self.scheduler.set_timesteps(num_steps)
        if init_img is not None:
            strength = float(np.clip(strength, 1.0 / num_steps, (num_steps - 1) / num_steps))
            t_start  = max(num_steps - int(num_steps * strength), 0)
            timesteps = self.scheduler.timesteps[t_start:]
            noise = torch.randn_like(init_latents)
            latents = self.scheduler.add_noise(init_latents, noise, timesteps[:1])
        else:
            timesteps = self.scheduler.timesteps
            latents = init_latents * self.scheduler.init_noise_sigma

        # 4. Prepare ControlNet condition image(s)
        depth_t  = self._cond_tensor(depth_img)   # [1, 3, H, W] in [0,1]
        if self.condition_mode == "both" and normal_img is not None:
            normal_t = self._cond_tensor(normal_img)
            cond_images = [depth_t, normal_t]
            cn_scales   = [cn_scale, cn_scale * 0.7]   # weight normal less
        elif self.condition_mode == "both":
            # Fallback: use depth for both slots
            cond_images = [depth_t, depth_t]
            cn_scales   = [cn_scale, cn_scale * 0.5]
        else:
            cond_images = depth_t
            cn_scales   = cn_scale

        # 5. Denoising loop
        self.controlnet.to(self.device)
        self.unet.to(self.device)

        # Duplicate cond for CFG (uncond + cond)
        if isinstance(cond_images, list):
            cond_images_cfg = [torch.cat([c, c]) for c in cond_images]
        else:
            cond_images_cfg = torch.cat([cond_images, cond_images])

        for t in timesteps:
            latent_in = torch.cat([latents] * 2)
            latent_in = self.scheduler.scale_model_input(latent_in, t)

            # ControlNet forward pass
            down_block_res, mid_block_res = self.controlnet(
                latent_in,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cond_images_cfg,
                conditioning_scale=cn_scales,
                return_dict=False,
            )

            # UNet forward pass with ControlNet residuals
            noise_pred = self.unet(
                latent_in,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            ).sample

            # CFG
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance * (cond - uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        _offload(self.controlnet, self.dtype)
        _offload(self.unet, self.dtype)

        return self._decode_latents(latents)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_texture(
        self,
        prompt: str,
        depth_collage_path: str,
        output_path: str,
        init_collage_path: str = None,
        normal_collage_path: str = None,
        num_steps: int = DEFAULT_STEPS,
        strength: float = DEFAULT_STRENGTH,
        cn_scale: float = DEFAULT_CN_SCALE,
        guidance: float = DEFAULT_GUIDANCE,
        neg_prompt: str = "",
        grid_mode: str = GRID_MODE,
    ):
        """
        Main entry point.

        Args:
            prompt              Texture description.
            depth_collage_path  Path to 3×2 depth-map collage (greyscale or RGB).
            output_path         Where to save the final textured collage.
            init_collage_path   Optional colour reference collage (img2img base).
            normal_collage_path Optional normal-map collage (used when cond_mode='both').
            num_steps           Denoising steps (20 is a good default).
            strength            img2img noise level (ignored if no init_collage_path).
            cn_scale            ControlNet conditioning strength.
            guidance            CFG scale.
            neg_prompt          Negative prompt.
            grid_mode           'collage' or 'tiles'.
        """
        if not os.path.exists(depth_collage_path):
            raise FileNotFoundError(depth_collage_path)

        depth_collage = Image.open(depth_collage_path).convert("RGB")
        init_collage  = Image.open(init_collage_path).convert("RGB") if init_collage_path else None
        norm_collage  = Image.open(normal_collage_path).convert("RGB") if normal_collage_path else None

        depth_collage = prepare_depth_map(depth_collage)   # ensure grey→RGB

        if grid_mode == "tiles":
            result = self._generate_tiles(
                prompt, neg_prompt,
                depth_collage, init_collage, norm_collage,
                num_steps, strength, cn_scale, guidance,
            )
        else:
            # Collage mode: resize whole image, one forward pass
            tw = _align(min(depth_collage.width, MAX_TILE_WIDTH * 3))
            th = _align(int(depth_collage.height * tw / depth_collage.width))
            depth_r = depth_collage.resize((tw, th), Image.LANCZOS)
            init_r  = init_collage.resize((tw, th), Image.LANCZOS) if init_collage else None
            norm_r  = norm_collage.resize((tw, th), Image.LANCZOS) if norm_collage else None
            result  = self._generate_single(
                prompt, neg_prompt, depth_r, init_r, norm_r,
                num_steps, strength, cn_scale, guidance,
            )

        result.save(output_path)
        print(f"[Done] Texture saved → {output_path}")
        return result

    def _generate_tiles(
        self, prompt, neg_prompt,
        depth_collage, init_collage, norm_collage,
        num_steps, strength, cn_scale, guidance,
        cols=3, rows=2,
    ) -> Image.Image:
        """Split, generate per-tile, re-stitch."""
        orig_w, orig_h = depth_collage.size
        tw_orig, th_orig = orig_w // cols, orig_h // rows

        d_tiles, _, _  = split_grid(depth_collage, cols, rows)
        i_tiles         = split_grid(init_collage, cols, rows)[0] if init_collage else [None] * cols * rows
        n_tiles         = split_grid(norm_collage,  cols, rows)[0] if norm_collage else [None] * cols * rows

        out_tiles = []
        for idx, (dt, it, nt) in enumerate(zip(d_tiles, i_tiles, n_tiles)):
            r, c = divmod(idx, cols)
            print(f"[Tile {idx+1}/{cols*rows}] row={r} col={c}")

            # Resize tile to align
            tile_w = _align(min(dt.width, MAX_TILE_WIDTH))
            tile_h = _align(int(dt.height * tile_w / dt.width))
            dt_r   = dt.resize((tile_w, tile_h), Image.LANCZOS)
            it_r   = it.resize((tile_w, tile_h), Image.LANCZOS) if it else None
            nt_r   = nt.resize((tile_w, tile_h), Image.LANCZOS) if nt else None

            out_tile = self._generate_single(
                prompt, neg_prompt, dt_r, it_r, nt_r,
                num_steps, strength, cn_scale, guidance,
            )
            out_tiles.append(out_tile)

        return stitch_grid(out_tiles, tw_orig, th_orig, cols, rows)


# ─── EXECUTION ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate textures using ControlNet.")
    parser.add_argument("--prompt", type=str, required=True, help="Texture description prompt")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth collage map")
    parser.add_argument("--output", type=str, required=True, help="Path for output texture map")
    parser.add_argument("--input", type=str, default=None, help="Optional normal/color init collage")
    parser.add_argument("--neg_prompt", type=str, default="cartoon, anime, painting, plastic, blurry, watermark, oversaturated, low resolution, jpeg artifacts")
    
    args = parser.parse_args()

    gen = ControlNetTextureGenerator(
        condition_mode=CONDITION_MODE,
    )

    gen.generate_texture(
        prompt=args.prompt,
        depth_collage_path=args.depth,
        output_path=args.output,
        init_collage_path=args.input,
        num_steps=DEFAULT_STEPS,
        strength=DEFAULT_STRENGTH,
        cn_scale=DEFAULT_CN_SCALE,
        guidance=DEFAULT_GUIDANCE,
        neg_prompt=args.neg_prompt,
        grid_mode=GRID_MODE,
    )