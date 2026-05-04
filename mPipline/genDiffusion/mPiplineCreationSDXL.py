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

# ─────────────────────────────────────────────────────────────────────────────
# Model paths  –  edit these to match your local layout
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PATHS = dict(
    base_model  = r"E:\Program Files\ComfyUI\ComfyUI\models\checkpoints\juggernautXL_v9Rdphoto2Lightning.safetensors",
    controlnet  = r"E:\Program Files\ComfyUI\ComfyUI\models\controlnet\diffusion_pytorch_model_promaxx.safetensors",

    depth_model = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\models\depth_anything_vitl14",
    vae = "",
)

# ── SDXL scheduler base config ───────────────────────────────────────────────
_SDXL_SCHEDULER_CONFIG = {
    "beta_end":               0.012,
    "beta_schedule":          "scaled_linear",
    "beta_start":             0.00085,
    "clip_sample":            False,
    "interpolation_type":     "linear",
    "num_train_timesteps":    1000,
    "prediction_type":        "epsilon",
    "sample_max_value":       1.0,
    "set_alpha_to_one":       False,
    "skip_prk_steps":         True,
    "steps_offset":           1,
    "timestep_spacing":       "leading",
    "use_karras_sigmas":      False,   # ComfyUI "normal" schedule
    "rescale_betas_zero_snr": False,
}


class ControlNetTextureGenerator:
    """
    End-to-end texture generator mirroring the ComfyUI workflow:

        LoadImage → ImageScale → DepthAnything → ControlNetApplySD3
        → KSampler(euler/normal) → VAEDecode → SaveImage

    All heavy models (SDXL checkpoint, ControlNet) load from raw .safetensors
    files using from_single_file(), so no diffusers-format folder is needed.

    Parameters
    ----------
    paths : dict | None
        Override any key in DEFAULT_PATHS.
    device : str
        'cuda', 'cpu', or 'mps'.
    dtype : torch.dtype
        torch.bfloat16 (default) or torch.float16.
    cpu_offload : bool
        Sequential CPU offload — very low VRAM usage, slower generation.
    compile_unet : bool
        torch.compile the UNet for faster repeated runs (PyTorch >= 2.0).
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

        print("[ControlNetTextureGenerator] Loading models ...")
        self._load_depth_model()
        self._load_pipeline()

        if compile_unet and hasattr(torch, "compile"):
            print("[ControlNetTextureGenerator] Compiling UNet ...")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )

        print("[ControlNetTextureGenerator] Ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Private loaders
    # ──────────────────────────────────────────────────────────────────────────

    def _load_depth_model(self):
        """
        Load Depth-Anything ViT-L14 for depth-map preprocessing.
        Mirrors ComfyUI node 55 (DepthAnythingPreprocessor).

        Expects a HuggingFace-style folder:
          depth_anything_vitl14/
            config.json
            preprocessor_config.json
            pytorch_model.bin  (or model.safetensors)
        Clone from: huggingface.co/LiheYoung/depth-anything-large-hf
        """
        depth_path = self.paths["depth_model"]
        print(f"  Depth-Anything  <- {depth_path}")

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_path)
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
            depth_path,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

    def _build_scheduler(self) -> EulerDiscreteScheduler:
        """
        Construct EulerDiscreteScheduler purely from a config dict.

        KEY FIX: never call .from_pretrained(path_to_safetensors) on a
        scheduler — diffusers tries to open the file as JSON and raises:
          OSError: It looks like the config file at '...' is not a valid JSON file.
        Instead we pass the config dict directly to the constructor.
        """
        return EulerDiscreteScheduler.from_config(_SDXL_SCHEDULER_CONFIG)

    def _load_pipeline(self):
        """
        Load the full SDXL + ControlNet pipeline from raw .safetensors files.

        Mirrors ComfyUI nodes:
          4  CheckpointLoaderSimple  -> from_single_file(base_model)
          46 ControlNetLoader        -> from_single_file(controlnet)
          56 KSampler(euler/normal)  -> EulerDiscreteScheduler from config dict
        """
        base  = self.paths["base_model"]
        cn    = self.paths["controlnet"]
        vae_p = self.paths["vae"]

        # ── ControlNet ────────────────────────────────────────────────────────
        print(f"  ControlNet      <- {cn}")
        controlnet = ControlNetModel.from_single_file(
            cn,
            torch_dtype=self.dtype,
        )

        # ── Optional standalone VAE ───────────────────────────────────────────
        vae_kwargs: dict = {}
        if vae_p:
            print(f"  VAE             <- {vae_p}")
            if vae_p.endswith(".safetensors"):
                vae_kwargs["vae"] = AutoencoderKL.from_single_file(
                    vae_p, torch_dtype=self.dtype
                )
            else:
                vae_kwargs["vae"] = AutoencoderKL.from_pretrained(
                    vae_p, torch_dtype=self.dtype
                )

        # ── SDXL pipeline from single .safetensors ────────────────────────────
        # from_single_file() reads binary weights only — it does NOT try to
        # parse the checkpoint as JSON, so the EnvironmentError is avoided.
        # Do NOT pass scheduler= here; set it after construction instead.
        print(f"  Base model      <- {base}")
        self.pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            base,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
            **vae_kwargs,
        )

        # ── Swap scheduler AFTER pipeline is built ────────────────────────────
        # from_single_file() attaches a default DDIMScheduler; replace it with
        # our Euler/normal scheduler built from the hardcoded config dict above.
        self.pipe.scheduler = self._build_scheduler()

        # ── Memory optimisations ──────────────────────────────────────────────
        self.pipe.enable_attention_slicing()

        if self.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Depth preprocessing  (mirrors nodes 48 + 55)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _depth_preprocess(
        self, image: Image.Image, resolution: int = 1024
    ) -> Image.Image:
        """
        Run Depth-Anything on `image` and return an RGB depth map.

        ComfyUI equivalent: DepthAnythingPreprocessor(resolution=1024)
        The longest side is resized to `resolution` before inference,
        then the depth map is upscaled back to the input dimensions.
        """
        w, h = image.size
        scale   = resolution / max(w, h)
        rw, rh  = int(w * scale), int(h * scale)
        resized = image.resize((rw, rh), Image.LANCZOS)

        inputs = self.depth_processor(images=resized, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        outputs  = self.depth_model(**inputs)
        depth    = outputs.predicted_depth          # (1, H, W)

        depth_np = depth.squeeze().cpu().float().numpy()
        lo, hi   = depth_np.min(), depth_np.max()
        depth_np = ((depth_np - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

        depth_pil = Image.fromarray(depth_np, mode="L").resize((w, h), Image.LANCZOS)
        return depth_pil.convert("RGB")             # ControlNet needs 3-channel

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_image: "str | Image.Image",
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
        seed: int             = 388481911638740,
        steps: int            = 30,
        cfg: float            = 8.0,
        # ── ControlNet strength (node 44) ──────────────────────────────────
        controlnet_strength: float = 1.0,
        # ── Size multiplier (nodes 61-66 Constant Number = 1.0) ───────────
        scale: float          = 1.0,
        # ── Depth-Anything resolution (node 55) ───────────────────────────
        depth_resolution: int = 1024,
    ) -> Image.Image:
        """
        Full pipeline: load -> scale -> depth -> ControlNet denoise -> save.

        Parameters
        ----------
        input_image
            Path string or PIL Image used as the depth source (node 45).
        output_path
            Destination file for the generated texture (node 9).
        positive_prompt / negative_prompt
            Text conditioning (nodes 68 / 67).
        seed
            Reproducibility seed (node 56 default: 388481911638740).
        steps
            Denoising steps (node 56 default: 30).
        cfg
            Classifier-free guidance scale (node 56 default: 8).
        controlnet_strength
            ControlNet conditioning weight (node 44 default: 1.0).
        scale
            Output resolution multiplier vs. input image (nodes 61-66).
        depth_resolution
            Max side length for Depth-Anything inference (node 55: 1024).

        Returns
        -------
        PIL.Image  -- also written to output_path.
        """
        # 1. Load ────────────────────────────────────────────────────────────
        if isinstance(input_image, str):
            img = Image.open(input_image).convert("RGB")
        else:
            img = input_image.convert("RGB")

        orig_w, orig_h = img.size

        # 2. Target size (nodes 61-66) -- snap to multiples of 64 ─────────────
        tgt_w = math.ceil(orig_w * scale / 64) * 64
        tgt_h = math.ceil(orig_h * scale / 64) * 64
        print(f"  Target size : {tgt_w} x {tgt_h}")

        # 3. Bilinear upscale (node 48 ImageScale) ────────────────────────────
        img_scaled = img.resize((tgt_w, tgt_h), Image.BILINEAR)

        # 4. Depth-Anything (node 55) ─────────────────────────────────────────
        print("  Running Depth-Anything ...")
        depth_map = self._depth_preprocess(img_scaled, resolution=depth_resolution)

        # 5. SDXL + ControlNet inference (nodes 44 + 56 + 8) ──────────────────
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"  Generating  : {steps} steps | cfg={cfg} | seed={seed}")
        result = self.pipe(
            prompt                        = positive_prompt,
            negative_prompt               = negative_prompt,
            image                         = depth_map,
            controlnet_conditioning_scale = controlnet_strength,
            width                         = tgt_w,
            height                        = tgt_h,
            num_inference_steps           = steps,
            guidance_scale                = cfg,
            generator                     = generator,
            num_images_per_prompt         = 1,
        )

        output_image: Image.Image = result.images[0]

        # 6. Save (node 9) ────────────────────────────────────────────────────
        output_image.save(output_path)
        print(f"  Saved -> {output_path}\n")

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return output_image


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Depth-ControlNet texture generator (ComfyUI workflow -> Python)"
    )
    p.add_argument("input_image", help="Reference image path (depth source)")
    p.add_argument("-o", "--output", default="output_texture.png")

    g = p.add_argument_group("Model paths")
    g.add_argument("--base-model",  default=DEFAULT_PATHS["base_model"])
    g.add_argument("--controlnet",  default=DEFAULT_PATHS["controlnet"])
    g.add_argument("--depth-model", default=DEFAULT_PATHS["depth_model"])
    g.add_argument("--vae",         default=DEFAULT_PATHS["vae"])

    g = p.add_argument_group("Generation")
    g.add_argument("--positive-prompt", default=None)
    g.add_argument("--negative-prompt", default=None)
    g.add_argument("--seed",        type=int,   default=3875984237444)
    g.add_argument("--steps",       type=int,   default=8)
    g.add_argument("--cfg",         type=float, default=6.0)
    g.add_argument("--cn-strength", type=float, default=1.0,  dest="cn_strength")
    g.add_argument("--scale",       type=float, default=1.0)
    g.add_argument("--depth-res",   type=int,   default=1024, dest="depth_res")

    g = p.add_argument_group("Hardware")
    g.add_argument("--device",      default="cuda", choices=["cuda", "cpu", "mps"])
    g.add_argument("--fp16",        action="store_true",
                   help="float16 instead of bfloat16 (for pre-Ampere GPUs)")
    g.add_argument("--cpu-offload", action="store_true",
                   help="Sequential CPU offload (very low VRAM, slow)")
    g.add_argument("--compile",     action="store_true",
                   help="torch.compile UNet for faster repeated runs")

    return p.parse_args()


def configuration():
    config = {
        "input_image": "in.png",  # required
        "output": "output_texture.png",

        "model_paths": {
            "base_model": DEFAULT_PATHS["base_model"],
            "controlnet": DEFAULT_PATHS["controlnet"],
            "depth_model": DEFAULT_PATHS["depth_model"],
            "vae": DEFAULT_PATHS["vae"],
        },

        "generation": {
            "positive_prompt": "8K ultra-detailed seamless texture of weathered rusted metal, industrial surface, realistic corrosion, peeling paint, scratches, grunge, high dynamic range, photorealistic, evenly lit, tileable, no seams",
            "negative_prompt": "low detail, blur, noise, distortion, watermark, text",
            "seed": 123456789012345,
            "steps": 12,
            "cfg": 4.0,
            "cn_strength": 1.0,
            "scale": 1.0,
            "depth_res": 1024,
        },

        "hardware": {
            "device": "cuda",  # options: "cuda", "cpu", "mps"
            "fp16": False,
            "cpu_offload": False,
            "compile": False,
        }
    }
    return config

def main():
    config = configuration()
    
    dtype = torch.float16 if config["hardware"]["fp16"] else torch.bfloat16

    gen = ControlNetTextureGenerator(
        paths        = dict(
            base_model  = config["model_paths"]["base_model"],
            controlnet  = config["model_paths"]["controlnet"],
            depth_model = config["model_paths"]["depth_model"],
            vae         = config["model_paths"]["vae"],
        ),
        device       = config["hardware"]["device"],
        dtype        = dtype,
        cpu_offload  = config["hardware"]["cpu_offload"],
        compile_unet = config["hardware"]["compile"],
    )

    kwargs: dict = dict(
        output_path         = config["output"],
        seed                = config["generation"]["seed"],
        steps               = config["generation"]["steps"],
        cfg                 = config["generation"]["cfg"],
        controlnet_strength = config["generation"]["cn_strength"],
        scale               = config["generation"]["scale"],
        depth_resolution    = config["generation"]["depth_res"],
    )
    if config["generation"]["positive_prompt"]:
        kwargs["positive_prompt"] = config["generation"]["positive_prompt"]
    if config["generation"]["negative_prompt"]:
        kwargs["negative_prompt"] = config["generation"]["negative_prompt"]

    gen.generate(config["input_image"], **kwargs)


def mainArgs():
    args = _parse_args()
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    gen = ControlNetTextureGenerator(
        paths        = dict(
            base_model  = args.base_model,
            controlnet  = args.controlnet,
            depth_model = args.depth_model,
            vae         = args.vae,
        ),
        device       = args.device,
        dtype        = dtype,
        cpu_offload  = args.cpu_offload,
        compile_unet = args.compile,
    )

    kwargs: dict = dict(
        output_path         = args.output,
        seed                = args.seed,
        steps               = args.steps,
        cfg                 = args.cfg,
        controlnet_strength = args.cn_strength,
        scale               = args.scale,
        depth_resolution    = args.depth_res,
    )
    if args.positive_prompt:
        kwargs["positive_prompt"] = args.positive_prompt
    if args.negative_prompt:
        kwargs["negative_prompt"] = args.negative_prompt

    gen.generate(args.input_image, **kwargs)

if __name__ == "__main__":
    mainArgs()