"""
ESRGAN Upscaler with CUDA support and model fine-tuning/refinement.

Dependencies:
    pip install torch torchvision basicsr facexlib gfpgan realesrgan Pillow tqdm

For CUDA support, install PyTorch with CUDA:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""

import sys

# sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

import torch
import os
import math
import time
from pathlib import Path
from typing import Optional

from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    _REALESRGAN_AVAILABLE = True
except ImportError:
    _REALESRGAN_AVAILABLE = False
    print("[WARNING] realesrgan not installed. Run: pip install realesrgan basicsr")


# ---------------------------------------------------------------------------
# Resolution presets (width x height)
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "8k":  (8192, 8192),
    "4k":  (4096, 4096),
    "2k":  (2048, 2048),
    "1k":  (1024, 1024),
    "512": (512, 512),
}


class ESRGANUpscaler:
    # Model registry: name -> (arch, scale, num_block, url)
    _MODEL_REGISTRY = {
        "RealESRGAN_x4plus": {
            "scale": 4,
            "num_block": 23,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "arch": "rrdbnet",
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "scale": 4,
            "num_block": 6,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "arch": "rrdbnet",
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        "RealESRGAN_x2plus": {
            "scale": 2,
            "num_block": 23,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "arch": "rrdbnet",
            "num_feat": 64,
            "num_grow_ch": 32,
        },
    }

    def __init__(
        self,
        input_images: list[str],
        output_folder: str,
        target_resolution: str | tuple[int, int] = "4k",
        model_name: str = "RealESRGAN_x4plus",
        device: Optional[str] = None,
        tile: int = 512,
        tile_pad: int = 32,
        half_precision: bool = True,
    ) -> None:
        self.input_images = [Path(p) for p in input_images]
        self.output_folder = Path(output_folder)
        self.model_name = model_name
        self.tile = tile
        self.tile_pad = tile_pad

        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.half_precision = half_precision and self.device.type == "cuda"

        # Resolve target resolution
        self.target_resolution = self._parse_resolution(target_resolution)

        # Infer base scale from model
        self._model_cfg = self._MODEL_REGISTRY.get(model_name)
        if self._model_cfg is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(self._MODEL_REGISTRY.keys())}"
            )

        self._upsampler: Optional[RealESRGANer] = None
        self._nn_model: Optional[nn.Module] = None

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._validate_inputs()
        self._print_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (if needed) and load the ESRGAN model onto device."""
        if not _REALESRGAN_AVAILABLE:
            raise RuntimeError("realesrgan package not installed.")

        cfg = self._model_cfg
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=cfg["num_feat"],
            num_block=cfg["num_block"],
            num_grow_ch=cfg["num_grow_ch"],
            scale=cfg["scale"],
        )
        model_path = str(self._get_model_path())

        self._upsampler = RealESRGANer(
            scale=cfg["scale"],
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=self.half_precision,
            device=self.device,
        )
        self._nn_model = self._upsampler.model
        print(f"[v] Model loaded on {self.device}  (fp{'16' if self.half_precision else '32'})")

    def upscale_all(self) -> list[Path]:
        """
        Upscale all input images to the target resolution.

        Returns
        -------
        list[Path]
            Paths to the saved output images.
        """
        if self._upsampler is None:
            self.load_model()

        results: list[Path] = []
        for img_path in tqdm(self.input_images, desc="Upscaling", unit="img"):
            out_path = self._process_image(img_path)
            if out_path:
                results.append(out_path)
        print(f"\n[v] Done. {len(results)}/{len(self.input_images)} images saved to '{self.output_folder}'")
        return results

    def refine_model(
        self,
        lr_images: list[str],
        hr_images: list[str],
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        save_refined_path: Optional[str] = None,
        loss_fn: str = "l1",
    ) -> None:

        if self._nn_model is None:
            self.load_model()

        assert len(lr_images) == len(hr_images), "LR and HR lists must be the same length."

        model = self._nn_model
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.L1Loss() if loss_fn == "l1" else nn.MSELoss()

        dataset = list(zip(lr_images, hr_images))
        save_path = save_refined_path or str(
            self._get_model_path().parent / f"{self.model_name}_refined.pth"
        )

        print(f"\n[*] Starting model refinement  ({epochs} epochs, lr={learning_rate})")
        print(f"    {len(dataset)} image pairs  |  loss={loss_fn.upper()}  |  device={self.device}")

        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            indices = np.random.permutation(len(dataset))

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                lr_tensors, hr_tensors = [], []

                for i in batch_idx:
                    lr_t = self._image_to_tensor(lr_images[i])
                    hr_t = self._image_to_tensor(hr_images[i])
                    lr_tensors.append(lr_t)
                    hr_tensors.append(hr_t)

                lr_batch = torch.cat(lr_tensors, dim=0).to(self.device)
                hr_batch = torch.cat(hr_tensors, dim=0).to(self.device)

                if self.half_precision:
                    lr_batch = lr_batch.half()
                    hr_batch = hr_batch.half()

                optimizer.zero_grad()
                sr_batch = model(lr_batch)

                # Crop HR to match SR spatial size (safety for boundary cases)
                _, _, sh, sw = sr_batch.shape
                hr_batch = hr_batch[:, :, :sh, :sw]

                loss = criterion(sr_batch, hr_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(1, math.ceil(len(dataset) / batch_size))

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({"params": model.state_dict()}, save_path)
                tag = " ← best"
            else:
                tag = ""

            print(f"  Epoch [{epoch:>3}/{epochs}]  loss={avg_loss:.6f}{tag}")

        print(f"\n[v] Refined model saved -> {save_path}")
        model.eval()

        # Reload best weights into upsampler
        state = torch.load(save_path, map_location=self.device)
        model.load_state_dict(state["params"], strict=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_image(self, img_path: Path) -> Optional[Path]:
        """Run ESRGAN on one image, then resize to exact target resolution."""
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)  # H x W x 3, uint8

            # ESRGAN inference (returns BGR numpy array)
            sr_np, _ = self._upsampler.enhance(img_np[:, :, ::-1], outscale=None)

            # Convert back to RGB PIL
            sr_img = Image.fromarray(sr_np[:, :, ::-1])

            # Resize to exact target if needed
            tw, th = self.target_resolution
            if sr_img.size != (tw, th):
                sr_img = self._smart_resize(sr_img, tw, th)

            out_path = self.output_folder / f"{img_path.stem}_upscaled.png"
            sr_img.save(out_path, format="PNG", optimize=False)
            return out_path

        except Exception as exc:
            print(f"\n[!] Failed to process '{img_path.name}': {exc}")
            return None

    def _smart_resize(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Resize to target, preserving aspect ratio with centre-crop.
        If the SR image is smaller than target, upscale with LANCZOS.
        """
        src_w, src_h = img.size
        scale = max(target_w / src_w, target_h / src_h)
        new_w = round(src_w * scale)
        new_h = round(src_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Centre crop
        left = (new_w - target_w) // 2
        top  = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        return img

    def _image_to_tensor(self, path: str) -> torch.Tensor:
        """Load image as normalised [0,1] float tensor (1 x C x H x W)."""
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
        return tensor

    def _get_model_path(self) -> Path:
        """Return local model path, downloading if absent."""
        weights_dir = Path.home() / ".cache" / "esrgan_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / f"{self.model_name}.pth"

        if not model_path.exists():
            print(f"[*] Downloading model weights -> {model_path}")
            import urllib.request
            url = self._model_cfg["url"]
            urllib.request.urlretrieve(url, model_path)
            print(f"[v] Download complete.")
        return model_path

    @staticmethod
    def _parse_resolution(res: str | tuple[int, int]) -> tuple[int, int]:
        if isinstance(res, tuple):
            return res
        key = res.lower().strip()
        if key in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[key]
        raise ValueError(
            f"Unknown resolution preset '{res}'. "
            f"Use one of {list(RESOLUTION_PRESETS.keys())} or pass (width, height)."
        )

    def _validate_inputs(self) -> None:
        missing = [p for p in self.input_images if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} input image(s) not found:\n  "
                + "\n  ".join(str(p) for p in missing)
            )
        non_png = [p for p in self.input_images if p.suffix.lower() != ".png"]
        if non_png:
            print(
                f"[!] {len(non_png)} file(s) are not .png — they will still be processed:\n  "
                + "\n  ".join(str(p) for p in non_png)
            )

    def _print_config(self) -> None:
        tw, th = self.target_resolution
        print("=" * 52)
        print("  ESRGAN Upscaler")
        print("=" * 52)
        print(f"  Device          : {self.device}")
        print(f"  Model           : {self.model_name}")
        print(f"  Target          : {tw}×{th} px")
        print(f"  Input images    : {len(self.input_images)}")
        print(f"  Output folder   : {self.output_folder}")
        print(f"  Tile size       : {self.tile or 'disabled'}")
        print(f"  Half precision  : {self.half_precision}")
        print("=" * 52)


def fastUpscaler(input_image_path = None, output_path = None):
    if input_image_path:
        upscaler = ESRGANUpscaler(
            input_images=[input_image_path],
            output_folder=output_path,
            target_resolution=(4096, 4096)
        )
        upscaler.load_model()
        results = upscaler.upscale_all()
        if results:
            return str(results[0])
    return None

    
# ---------------------------------------------------------------------------
# Quick-start usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESRGAN Upscaler CLI")
    parser.add_argument("--input", nargs="+", required=True, help="Path(s) to input images")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--res", default="2k", help="Target resolution (e.g., 2k, 4k)")
    parser.add_argument("--model", default="RealESRGAN_x4plus", help="Model name")
    
    args = parser.parse_args()

    upscaler = ESRGANUpscaler(
        input_images=args.input,
        output_folder=args.output,
        target_resolution=args.res,
        model_name=args.model,
        tile=512,
        half_precision=True,
    )

    upscaler.load_model()
    upscaler.upscale_all()