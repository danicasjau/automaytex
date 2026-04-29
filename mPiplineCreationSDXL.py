"""
ControlNetTextureGenerator
==========================
Standalone Python script converted from a ComfyUI workflow that:
  1. Loads a depth image (e.g. depth_retarget_1001.png)
  2. Runs Depth-Anything preprocessing
  3. Applies ControlNet (depth) conditioning
  4. Generates a seamless texture with JuggernautXL + KSampler

ComfyUI node → diffusers mapping
─────────────────────────────────
CheckpointLoaderSimple    → UNet2DConditionModel + CLIPTokenizer/CLIPTextModel + AutoencoderKL
DepthAnythingPreprocessor → transformers DepthAnythingForDepthEstimation (vitl)
ControlNetApplySD3        → ControlNetModel + StableDiffusionXLControlNetPipeline
KSampler (euler/normal)   → EulerDiscreteScheduler
VAEDecode / SaveImage     → vae.decode + PIL save

GPU memory strategy
────────────────────
• BF16 weights (bfloat16) — native Ampere/Ada precision, ~2× smaller than FP32
• torch.compile (optional, toggled via flag) — fused CUDA kernels
• Sequential CPU offload (optional) — streams layers to GPU one at a time
• Attention slicing — limits peak VRAM per attention op
• No xformers dependency (uses PyTorch 2.x SDPA instead)
"""

import sys, os, gc, math, argparse
import numpy as np
import torch
from PIL import Image, ImageFilter

# ── Allow a local venv to shadow system packages ──────────────────────────────
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,       # SDXL needs the second text encoder
    DepthAnythingForDepthEstimation,   # Depth-Anything vision backbone
    AutoImageProcessor,
)

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,            # "euler / normal" scheduler from ComfyUI
    UNet2DConditionModel,
)
from diffusers.utils import load_image


# ─────────────────────────────────────────────────────────────────────────────
# Model paths  –  edit these to match your local layout
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PATHS = dict(
    # JuggernautXL checkpoint directory (converted to diffusers layout)
    # or the original .safetensors – diffusers can load single-file SDXL ckpts
    base_model    = r"E:\Program Files\ComfyUI\ComfyUI\models\checkpoints\juggernautXL_v9Rdphoto2Lightning.safetensors",

    # ControlNet for SDXL depth (promaxx variant)
    controlnet    = r"E:\Program Files\ComfyUI\ComfyUI\models\controlnet\diffusion_pytorch_model_promaxx.safetensors",

    # Depth-Anything ViT-L  (HuggingFace-style repo or local folder)
    depth_model   = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\models\depth_anything_vitl14",

    # Optional: standalone VAE (leave empty to use the one inside base_model)
    vae           = "",
)

# ─────────────────────────────────────────────────────────────────────────────

class ControlNetTextureGenerator:
    """
    End-to-end texture generator that mirrors the ComfyUI workflow:

        LoadImage → ImageScale → DepthAnything → ControlNetApplySD3
        → KSampler(euler) → VAEDecode → SaveImage

    Parameters
    ----------
    paths : dict
        Override any key in DEFAULT_PATHS.
    device : str
        'cuda', 'cpu', or 'mps'.
    dtype : torch.dtype
        torch.bfloat16 (default) or torch.float16.
    cpu_offload : bool
        Stream model layers to CPU when not needed (slow but low-VRAM).
    compile_unet : bool
        Run torch.compile on the UNet (faster after warm-up, needs PyTorch ≥ 2.0).
    """

    def __init__(
        self,
        paths: dict | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False,
        compile_unet: bool = False,
    ):
        self.paths       = {**DEFAULT_PATHS, **(paths or {})}
        self.device      = torch.device(device)
        self.dtype       = dtype
        self.cpu_offload = cpu_offload

        print("[ControlNetTextureGenerator] Loading models …")
        self._load_depth_model()
        self._load_pipeline()

        if compile_unet and hasattr(torch, "compile"):
            print("[ControlNetTextureGenerator] Compiling UNet with torch.compile …")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )

        print("[ControlNetTextureGenerator] Ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_depth_model(self):
        """Load Depth-Anything ViT-L14 for preprocessing (mirrors node 55)."""
        depth_path = self.paths["depth_model"]
        print(f"  • Depth-Anything  ← {depth_path}")

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_path)
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
            depth_path,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.depth_model.eval()

    def _load_pipeline(self):
        """
        Build the SDXL + ControlNet pipeline.

        Mirrors ComfyUI nodes:
          4  CheckpointLoaderSimple  → base UNet / CLIP / VAE
          46 ControlNetLoader        → ControlNetModel
          56 KSampler(euler/normal)  → EulerDiscreteScheduler
        """
        base   = self.paths["base_model"]
        cn     = self.paths["controlnet"]
        vae_p  = self.paths["vae"]

        print(f"  • ControlNet      ← {cn}")
        # Single-file .safetensors ControlNet
        if cn.endswith(".safetensors"):
            controlnet = ControlNetModel.from_single_file(
                cn, torch_dtype=self.dtype
            )
        else:
            controlnet = ControlNetModel.from_pretrained(
                cn, torch_dtype=self.dtype
            )

        # Scheduler: euler + normal (linear beta schedule) ── node 56
        scheduler = EulerDiscreteScheduler.from_pretrained(
            base, subfolder="scheduler"
        )
        scheduler.config.use_karras_sigmas = False   # "normal" schedule

        # Optional standalone VAE (node 4 uses the checkpoint's built-in one)
        vae_kwargs = {}
        if vae_p:
            print(f"  • VAE             ← {vae_p}")
            vae_kwargs["vae"] = AutoencoderKL.from_pretrained(
                vae_p, torch_dtype=self.dtype
            )

        print(f"  • Base model      ← {base}")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base,
            controlnet=controlnet,
            scheduler=scheduler,
            torch_dtype=self.dtype,
            variant="fp16",          # load fp16 weights if present; falls back gracefully
            use_safetensors=True,
            **vae_kwargs,
        )

        # Memory optimisations
        self.pipe.enable_attention_slicing()         # cap VRAM per attention op

        if self.cpu_offload:
            # Streams every sub-module to GPU only when needed
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Depth preprocessing  (mirrors nodes 48 + 55)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _depth_preprocess(self, image: Image.Image, resolution: int = 1024) -> Image.Image:
        """
        Run Depth-Anything on `image` and return a grayscale depth map.

        ComfyUI equivalent: DepthAnythingPreprocessor(resolution=1024)
        """
        # Resize longest side to `resolution` before inference (node 55 param)
        w, h = image.size
        scale = resolution / max(w, h)
        rw, rh = int(w * scale), int(h * scale)
        img_resized = image.resize((rw, rh), Image.LANCZOS)

        inputs = self.depth_processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        outputs = self.depth_model(**inputs)
        depth   = outputs.predicted_depth  # (1, H, W)  float32

        # Normalise to [0, 255] and resize back to the requested output size
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        depth_np = (depth_np * 255).astype(np.uint8)

        depth_pil = Image.fromarray(depth_np, mode="L").resize((w, h), Image.LANCZOS)
        return depth_pil.convert("RGB")          # ControlNet expects 3-channel

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_image: str | Image.Image,
        output_path: str = "output_texture.png",
        # ── Prompts (nodes 67 / 68) ────────────────────────────────────────
        positive_prompt: str = (
            "8K ultra-detailed seamless texture of weathered aged wood, "
            "industrial surface, realistic grain patterns, cracks, peeling paint, "
            "scratches, grunge, high dynamic range, photorealistic, evenly lit, "
            "tileable, no seams"
        ),
        negative_prompt: str = (
            "blurry, low quality, distorted, shadow, lighting gradients"
        ),
        # ── KSampler params (node 56) ──────────────────────────────────────
        seed: int           = 388481911638740,
        steps: int          = 30,
        cfg: float          = 8.0,
        # ── ControlNet params (node 44) ────────────────────────────────────
        controlnet_strength: float = 1.0,
        # ── Output size scale (nodes 61–66) – 1.0 = original image size ───
        scale: float        = 1.0,
        # ── Depth preprocessing ────────────────────────────────────────────
        depth_resolution: int = 1024,
    ) -> Image.Image:
        """
        Full pipeline: depth-preprocess → ControlNet → denoise → decode → save.

        Parameters
        ----------
        input_image : path or PIL Image
            The reference image used to generate the depth map (node 45 LoadImage).
        output_path : str
            Where to write the result (node 9 SaveImage).
        positive_prompt, negative_prompt
            Text conditioning (nodes 68 / 67).
        seed : int
            RNG seed (node 56).
        steps : int
            Denoising steps (node 56, default 30).
        cfg : float
            Classifier-free guidance scale (node 56, default 8).
        controlnet_strength : float
            ControlNet conditioning scale (node 44 strength, default 1.0).
        scale : float
            Multiply original image dimensions (nodes 61–66 Constant Number).
        depth_resolution : int
            Max resolution fed to Depth-Anything (node 55, default 1024).

        Returns
        -------
        PIL.Image  (also saved to output_path)
        """
        # ── 1. Load image (node 45) ─────────────────────────────────────────
        if isinstance(input_image, str):
            img = Image.open(input_image).convert("RGB")
        else:
            img = input_image.convert("RGB")

        orig_w, orig_h = img.size

        # ── 2. Compute target size (nodes 61–66) ────────────────────────────
        tgt_w = int(orig_w * scale)
        tgt_h = int(orig_h * scale)
        # Make dimensions multiples of 64 (VAE / latent grid requirement)
        tgt_w = math.ceil(tgt_w / 64) * 64
        tgt_h = math.ceil(tgt_h / 64) * 64
        print(f"  Target size: {tgt_w} × {tgt_h}")

        # ── 3. ImageScale (node 48) ─────────────────────────────────────────
        img_scaled = img.resize((tgt_w, tgt_h), Image.BILINEAR)

        # ── 4. Depth-Anything preprocess (node 55) ──────────────────────────
        print("  Running Depth-Anything …")
        depth_map = self._depth_preprocess(img_scaled, resolution=depth_resolution)

        # ── 5. Run SDXL + ControlNet pipeline (nodes 44 + 56 + 8) ──────────
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"  Generating ({steps} steps, cfg={cfg}, seed={seed}) …")
        result = self.pipe(
            prompt                  = positive_prompt,
            negative_prompt         = negative_prompt,
            image                   = depth_map,          # ControlNet image input
            controlnet_conditioning_scale = controlnet_strength,
            width                   = tgt_w,
            height                  = tgt_h,
            num_inference_steps     = steps,
            guidance_scale          = cfg,
            generator               = generator,
            num_images_per_prompt   = 1,
            # node 56: denoise = 1.0  ↔ full denoising (text-to-image mode)
            denoising_end           = 1.0,
        )

        output_image: Image.Image = result.images[0]

        # ── 6. Save (node 9) ────────────────────────────────────────────────
        output_image.save(output_path)
        print(f"  ✓ Saved → {output_path}")

        # Free intermediate tensors
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return output_image


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Depth-ControlNet texture generator (ComfyUI workflow → Python)"
    )
    p.add_argument("input_image",  help="Path to the reference image (depth source)")
    p.add_argument("-o", "--output", default="output_texture.png",
                   help="Output file path (default: output_texture.png)")

    # Model paths
    p.add_argument("--base-model",  default=DEFAULT_PATHS["base_model"])
    p.add_argument("--controlnet",  default=DEFAULT_PATHS["controlnet"])
    p.add_argument("--depth-model", default=DEFAULT_PATHS["depth_model"])
    p.add_argument("--vae",         default=DEFAULT_PATHS["vae"])

    # Generation
    p.add_argument("--positive-prompt", default=None,
                   help="Override the positive text prompt")
    p.add_argument("--negative-prompt", default=None,
                   help="Override the negative text prompt")
    p.add_argument("--seed",    type=int,   default=388481911638740)
    p.add_argument("--steps",   type=int,   default=30)
    p.add_argument("--cfg",     type=float, default=8.0)
    p.add_argument("--cn-strength", type=float, default=1.0,
                   dest="cn_strength")
    p.add_argument("--scale",   type=float, default=1.0,
                   help="Output size multiplier relative to input (default: 1.0)")
    p.add_argument("--depth-res", type=int, default=1024,
                   dest="depth_res",
                   help="Max resolution for Depth-Anything (default: 1024)")

    # Hardware
    p.add_argument("--device",       default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--fp16",         action="store_true",
                   help="Use float16 instead of bfloat16")
    p.add_argument("--cpu-offload",  action="store_true",
                   help="Enable sequential CPU offload (very low VRAM)")
    p.add_argument("--compile",      action="store_true",
                   help="torch.compile the UNet (faster after first run)")

    return p.parse_args()


def deafult_config_args():
    pass


def main():
    args = _parse_args()

    paths = dict(
        base_model  = args.base_model,
        controlnet  = args.controlnet,
        depth_model = args.depth_model,
        vae         = args.vae,
    )

    dtype = torch.float16 if args.fp16 else torch.bfloat16

    gen = ControlNetTextureGenerator(
        paths       = paths,
        device      = args.device,
        dtype       = dtype,
        cpu_offload = args.cpu_offload,
        compile_unet= args.compile,
    )

    kwargs = dict(
        seed               = args.seed,
        steps              = args.steps,
        cfg                = args.cfg,
        controlnet_strength= args.cn_strength,
        scale              = args.scale,
        depth_resolution   = args.depth_res,
        output_path        = args.output,
    )
    if args.positive_prompt:
        kwargs["positive_prompt"] = args.positive_prompt
    if args.negative_prompt:
        kwargs["negative_prompt"] = args.negative_prompt

    gen.generate(args.input_image, **kwargs)


if __name__ == "__main__":
    main()