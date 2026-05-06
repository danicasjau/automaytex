import sys, os, gc, math, argparse
import numpy as np
import torch  # type: ignore
from PIL import Image

# ── Allow a local venv to shadow system packages ──────────────────────────────
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

from transformers import (  # type: ignore
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    DepthAnythingForDepthEstimation,
    AutoImageProcessor,
)
from diffusers import (  # type: ignore
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)

class DiffGenSDXL:
    def __init__(self, cmodels, config):

        self.models = cmodels

        self.device = cmodels.get_device()
        self.dtype  = cmodels.get_dtype()

        # ----------------------------
        # SET INITIAL CONFIGURATION
        # ----------------------------
        self.input_image = config["input_image"]
        self.output_path = config["output_path"]

        self.positive_prompt = config["positive_prompt"]
        self.negative_prompt = config["negative_prompt"]

        self.seed = config["seed"] or 388481911638740
        self.steps = config["steps"] or 30
        self.cfg = config["cfg"] or 8.0
        self.controlnet_strength = config["controlnet_strength"] or 1.0

        self.scale = config["scale"] or 1.0
        self.depth_resolution = config["depth_resolution"] or 1024

        # ----------------------------
        # LOAD MODELS
        # ----------------------------

        compile_unet = False

        if compile_unet and hasattr(torch, "compile"):
            print("[ControlNetTextureGenerator] Compiling UNet deferred...")
            # Will be handled during generation if enabled

        print("[ControlNetTextureGenerator] Ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Depth preprocessing  (mirrors nodes 48 + 55)
    # ──────────────────────────────────────────────────────────────────────────


    @torch.inference_mode()
    def _depth_preprocess(self, image: Image.Image, resolution: int = 1024):
        """
        Run Depth-Anything on `image` and return an RGB depth map.
        """
        w, h = image.size
        scale   = resolution / max(w, h)
        rw, rh  = int(w * scale), int(h * scale)
        resized = image.resize((rw, rh), Image.LANCZOS)

        #inputs = self.depth_processor(images=resized, return_tensors="pt")
        inputs = self.models.depth_processor(images=resized, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        outputs  = self.models.depth_model(**inputs)
        depth    = outputs.predicted_depth          # (1, H, W)

        depth_np = depth.squeeze().cpu().float().numpy()
        lo, hi   = depth_np.min(), depth_np.max()
        depth_np = ((depth_np - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

        depth_pil = Image.fromarray(depth_np, mode="L").resize((w, h), Image.LANCZOS)
        return depth_pil.convert("RGB")             # ControlNet needs 3-channel

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self):
        if self.models.pipe is None or self.models.depth_processor is None:
            print("[DiffGenSDXL] Models not loaded. Loading now...")
            self.models.load_all()

        # ------------------------------------------------------------------
        # Resolve output file path — config sends a folder, PIL needs a file
        # ------------------------------------------------------------------
        out_path = self.output_path
        ext = os.path.splitext(out_path)[1].lower()
        if not ext:
            # output_path is a directory – append a filename
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, "diffuse.png")
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with torch.inference_mode():
            # 1. Load ─────────────────────────────────────────────────────
            if isinstance(self.input_image, str):
                img = Image.open(self.input_image).convert("RGB")
            else:
                img = self.input_image.convert("RGB")

            orig_w, orig_h = img.size

            # 2. Target size – snap to multiples of 64 ────────────────────
            tgt_w = math.ceil(orig_w * self.scale / 64) * 64
            tgt_h = math.ceil(orig_h * self.scale / 64) * 64
            print(f"  Target size : {tgt_w} x {tgt_h}")

            # 3. Bilinear upscale ─────────────────────────────────────────
            img_scaled = img.resize((tgt_w, tgt_h), Image.BILINEAR)

            # 4. Depth-Anything ───────────────────────────────────────────
            print("  Running Depth-Anything ...")
            depth_map = self._depth_preprocess(img_scaled, resolution=self.depth_resolution)

            # 5. SDXL + ControlNet ────────────────────────────────────────
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            print(f"  Generating  : {self.steps} steps | cfg={self.cfg} | seed={self.seed}")
            result = self.models.pipe(
                prompt                        = self.positive_prompt,
                negative_prompt               = self.negative_prompt,
                image                         = depth_map,
                controlnet_conditioning_scale = self.controlnet_strength,
                width                         = tgt_w,
                height                        = tgt_h,
                num_inference_steps           = self.steps,
                guidance_scale                = self.cfg,
                generator                     = generator,
                num_images_per_prompt         = 1,
            )

        output_image: Image.Image = result.images[0]

        # 6. Save ─────────────────────────────────────────────────────────
        output_image.save(out_path)
        print(f"  Saved -> {out_path}\n")

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return out_path   # return the actual file path, not the PIL image
