# automaytex\cModels.py
# models control
# automaytex/cModels.py
# Model management: load, unload, install diffusion + controlnet + depth models


import os
import json
import torch

import requests
from pathlib import Path
from tqdm import tqdm  # type: ignore

from transformers import (  # type: ignore
    DepthAnythingForDepthEstimation,
    AutoImageProcessor,
)

from diffusers import (  # type: ignore
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
)


# ---------------------------------------------------------------------------
# Quantization dtype map
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    None:   torch.float32,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # int8 / int4 require bitsandbytes – kept as fp16 fallback here;
    # swap in BitsAndBytesConfig if bitsandbytes is available
    "int8": torch.float16,
    "int4": torch.float16,
}

SUPPORTED_BASE_MODELS = ["sdxl", "sd15", "fast_sdxl", "flash_sdxl"]
SUPPORTED_QUANTIZATIONS = [None, "fp16", "int8", "int4", "bf16", "fp32"]

# ---------------------------------------------------------------------------
# cModels
# ---------------------------------------------------------------------------
class diffModels:
    def __init__(self, configuration=None):
        if configuration is None:
            from config import configuration as _cfg  # type: ignore
            configuration = _cfg()

        self.config = configuration

        # Resolve torch dtype from quantization string
        quant = getattr(self.config, "quantization", "fp16")
        self.dtype = DTYPE_MAP.get(quant, torch.float16)

        # Resolve compute device
        self.device = self._resolve_device()

        # CPU offload flag (moves layers to CPU between forward passes)
        self.cpu_offload = getattr(self.config, "cpu_offload", False)

        self.diffusion_model_type = getattr(self.config, "base_model", "flash_sdxl")

        # Public handles – None until load_all() is called
        self.diffusion_model = None   # StableDiffusionXLControlNetPipeline
        self.depth_model      = None   # DepthAnythingForDepthEstimation
        self.depth_processor  = None   # AutoImageProcessor
        self.pipe             = None   # alias → same as diffusion_model

        # Load the models catalogue from JSON
        self.catalogue = self._load_catalogue()

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------
    def _resolve_device(self):
        preferred = getattr(self.config, "preferred_device", "gpu").lower()
        cuda_ok = torch.cuda.is_available()

        if preferred in ("gpu", "cuda", "both"):
            if cuda_ok:
                device = torch.device("cuda")
                print(f"[cModels] CUDA available – using GPU  "
                      f"({torch.cuda.get_device_name(0)})")
            else:
                device = torch.device("cpu")
                print("[cModels] CUDA not available – falling back to CPU")
        else:
            device = torch.device("cpu")
            print("[cModels] Preferred device: CPU")

        return device

    # ------------------------------------------------------------------
    # JSON catalogue
    # ------------------------------------------------------------------
    def _load_catalogue(self) -> dict:
        json_path = getattr(self.config, "models_json", "models.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"[cModels] models.json not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        catalogue = {entry["name"]: entry for entry in raw.get("models", [])}
        print(f"[cModels] Catalogue loaded – {len(catalogue)} entries: "
              f"{list(catalogue.keys())}")
        return catalogue

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _get_local_path(self, name: str) -> str:
        entry = self.catalogue[name]
        return os.path.join(entry["installation_path"], entry["installation_name"])

    # ------------------------------------------------------------------
    # Optional model installation
    # ------------------------------------------------------------------
    def install_models(self):
        if not getattr(self.config, "installIfMissing", False):
            print("[cModels] installIfMissing=False – skipping installation")
            return

        models_dir = getattr(self.config, "models_directory", "models/")

        for name, entry in self.catalogue.items():
            local_path = self._get_local_path(name)

            # Directories (e.g. depth model folder) – check for model file inside
            if os.path.isdir(local_path) or os.path.isfile(local_path):
                print(f"[cModels] ✓ Already present: {name}")
                continue

            url  = entry["hugging_face_url"]
            dest_dir  = entry["installation_path"]
            dest_name = entry["download_name"]
            dest_full = os.path.join(dest_dir, dest_name)

            os.makedirs(dest_dir, exist_ok=True)
            print(f"[cModels] ↓ Downloading {name} from:\n  {url}")
            self._download_file(url, dest_full)

    def _download_file(self, url: str, dest: str):
        """Stream-download a file with a progress bar."""
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))

        print(f"[cModels]   Saved → {dest}")

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------
    def _load_controlnet(self, path: str) -> ControlNetModel:
        print(f"[cModels] Loading ControlNet from:\n  {path}")
        return ControlNetModel.from_single_file(
            path,
            torch_dtype=self.dtype,
        )

    def _load_vae(self, path: str) -> AutoencoderKL | None:
        """Return an AutoencoderKL or None if no path is given."""
        if not path:
            return None
        print(f"[cModels] Loading VAE from:\n  {path}")
        if path.endswith(".safetensors"):
            return AutoencoderKL.from_single_file(path, torch_dtype=self.dtype)
        return AutoencoderKL.from_pretrained(path, torch_dtype=self.dtype)

    def _load_sdxl_pipeline(self,base_path: str, controlnet: ControlNetModel, vae=None):
        print(f"[cModels] Loading SDXL pipeline from:\n  {base_path}")

        vae_kwargs = {"vae": vae} if vae is not None else {}

        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            base_path,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
            **vae_kwargs,
        )

        # --- Custom pipeline settings ---
        pipe.enable_attention_slicing()

        # Speed: channels_last memory layout (faster CUDA convolutions)
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            print("[cModels] channels_last memory format enabled")
        except Exception:
            pass

        # Speed: xformers memory-efficient attention (if installed)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[cModels] xformers memory-efficient attention enabled")
        except Exception:
            print("[cModels] xformers not available – using default attention")

        if self.cpu_offload:
            print("[cModels] Enabling sequential CPU offload")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)

        return pipe

    def _load_depth_model(self, path: str):
        """Load DepthAnything processor + model."""
        print(f"[cModels] Loading Depth model from:\n  {path}")
        processor = AutoImageProcessor.from_pretrained(path)
        model     = DepthAnythingForDepthEstimation.from_pretrained(
            path, torch_dtype=self.dtype
        ).to(self.device)
        return processor, model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_model_data(self, name: str) -> dict:
        """Return the catalogue entry for a model by name."""
        if name not in self.catalogue:
            raise KeyError(f"[cModels] Unknown model '{name}' in catalogue")
        return self.catalogue[name]

    def get_device(self):
        return self.device

    def get_dtype(self):
        return self.dtype

    def load_all(self, configuration=None):
        # 1. Install missing models (no-op if flag is False)
        self.install_models()

        if configuration is not None:
            print("[cModels] UPDATING CONFIGURATION")
            self.config = configuration

        # 2. Validate base model choice
        base_model_name = self.config["base_model"]
        self.diffusion_model_type = base_model_name
        self.quantization = self.config["quantization"]

        if base_model_name not in SUPPORTED_BASE_MODELS:
            raise ValueError(
                f"[cModels] Unsupported base_model '{base_model_name}'. "
                f"Choose from {SUPPORTED_BASE_MODELS}"
            )

        # 3. Resolve local paths from catalogue
        base_path       = self._get_local_path(self.diffusion_model_type) # "sdxl")        # always SDXL here
        controlnet_path = self._get_local_path("controlnet")
        depth_path      = self._get_local_path("depth")

        print(f"""
        #######################################################################
        #######################################################################
        LOADING MODEL {self.diffusion_model_type}

        #######################################################################

        Quantization: {self.quantization}
        Device: {self.device}

        -----------------------------------------------------------------------
        model path: {base_path}
        controlnet_path: {controlnet_path}
        depth_path: {depth_path}

        #######################################################################
        #######################################################################
        """)


        # vae_path        = getattr(self.config, "vae_path", "")  # optional

        print(f"\n[cModels] === Loading all models (quantization={self.quantization}, "
              f"device={self.device}) ===")

        # 4. ControlNet
        controlnet = self._load_controlnet(controlnet_path)

        # 5. VAE (optional)
        # vae = self._load_vae(vae_path)

        # 6. Diffusion pipeline
        if base_model_name == "sdxl":
            self.pipe = self._load_sdxl_pipeline(base_path, controlnet)#, vae)
        elif base_model_name == "fast_sdxl":
            self.pipe = self._load_sdxl_pipeline(base_path, controlnet)#, vae)
        elif base_model_name == "flash_sdxl":
            self.pipe = self._load_sdxl_pipeline(base_path, controlnet)#, vae)
        elif base_model_name == "sd15":
            # SD1.5 branch – extend here with StableDiffusionControlNetPipeline
            raise NotImplementedError("[cModels] SD1.5 pipeline not yet implemented")

        self.diffusion_model = self.pipe   # public alias

        # 7. Depth model
        self.depth_processor, self.depth_model = self._load_depth_model(depth_path)

        print("[cModels] === All models loaded ===\n")

    def unload_all(self):
        print("[cModels] Unloading all models …")

        self.pipe             = None
        self.diffusion_model  = None
        self.depth_model      = None
        self.depth_processor  = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("[cModels] All models unloaded – VRAM/RAM released")