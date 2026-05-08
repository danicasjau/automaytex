#!/usr/bin/env python3
"""
Multiview Image Generation Script
Converted from ComfyUI workflow (multiviewAPI.json)

Pipeline:
  1. Load SDXL checkpoint (JuggernautXL)
  2. Apply IPAdapter (style transfer from reference image)
  3. Apply ControlNet x2 (depth + normals collages)
  4. KSampler → VAEDecode → Save

Optimized for 8 GB VRAM via:
  - torch.float16 throughout
  - Sequential CPU offload
  - Attention slicing + xformers (if available)
  - VAE tiling

Usage:
    python multiview_generate.py \
        --checkpoint   /path/to/juggernautXL_v9Rdphoto2Lightning.safetensors \
        --controlnet   /path/to/diffusion_pytorch_model_promaxx.safetensors \
        --ipadapter    /path/to/ip-adapter_sdxl.safetensors \
        --clip_vision  /path/to/clip_vision_g.safetensors \
        --ref_image    /path/to/reference.png \
        --depth_image  /path/to/collage_depth.png \
        --normals_image /path/to/collage_normals.png \
        --output       /path/to/output.png \
        [--seed 871128783595447] \
        [--steps 30] \
        [--cfg 9.0] \
        [--width 1024] \
        [--height 1024]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_image_as_tensor(path: str, width: int, height: int) -> torch.Tensor:
    """Load image → resize → return (1, H, W, C) float32 tensor in [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)          # (1, H, W, 3)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1, H, W, C) float tensor [0,1] → PIL Image."""
    arr = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


# ──────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────

def load_sdxl_pipeline(checkpoint_path: str, device: torch.device):
    """
    Load SDXL UNet + VAE + CLIP from a single .safetensors checkpoint.
    Falls back to diffusers StableDiffusionXLPipeline if needed.
    """
    from diffusers import StableDiffusionXLPipeline

    print(f"[1/5] Loading SDXL checkpoint: {checkpoint_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # ── VRAM optimisations ──────────────────────
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  ✓ xformers memory-efficient attention enabled")
    except Exception:
        print("  ⚠ xformers not available, using PyTorch SDPA")
        pipe.unet.set_attn_processor(
            __import__("diffusers").models.attention_processor.AttnProcessor2_0()
        )

    # Sequential CPU offload keeps only the active module on GPU
    pipe.enable_sequential_cpu_offload(gpu_id=0)
    print("  ✓ Sequential CPU offload enabled (8 GB VRAM mode)")

    return pipe


def load_controlnet(controlnet_path: str, device: torch.device):
    """Load a ControlNet from a .safetensors file (SDXL-compatible)."""
    from diffusers import ControlNetModel

    print(f"[2/5] Loading ControlNet: {controlnet_path}")
    cn = ControlNetModel.from_single_file(
        controlnet_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    return cn


def load_ipadapter(ipadapter_path: str, clip_vision_path: str, pipe, device: torch.device):
    """
    Load IPAdapter weights and inject them into the pipeline.
    Uses ip_adapter library (install: pip install ip-adapter).
    """
    try:
        from ip_adapter import IPAdapterXL
    except ImportError:
        print("  ⚠ ip-adapter package not found. Install with:")
        print("      pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
        sys.exit(1)

    print(f"[3/5] Loading IPAdapter: {ipadapter_path}")
    ip_model = IPAdapterXL(
        pipe,
        image_encoder_path=str(Path(clip_vision_path).parent),   # folder containing model
        ip_ckpt=ipadapter_path,
        device=str(device),
    )
    return ip_model


# ──────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────

POSITIVE_PROMPT = (
    "A single wooden monkey sculpture shown in a 2x2 grid multi-view layout, "
    "each panel showing the same identical object with perfectly consistent geometry and texture. "
    "Top-left: front view. Top-right: right side view. Bottom-left: back view. Bottom-right: left side view. "
    "The sculpture is ultra-realistic carved wood, with highly detailed natural wood grain that flows consistently "
    "across all views, visible knots and fine grain direction continuity. Sharp carved edges, smooth polished surfaces, "
    "physically accurate material. "
    "Soft studio lighting, neutral gray background, global illumination, subtle contact shadows. "
    "Perspective matched across views, same scale and alignment, orthographic-style camera, no distortion. "
    "Photorealistic, 8k, high detail, consistent identity across all views, no variation between panels except camera angle."
)

NEGATIVE_PROMPT = (
    "different objects, inconsistent texture, mismatched grain, warped geometry, changing shape, "
    "different lighting per panel, different scale, distortion, blur, noise, low detail, stylized, cartoon, painting"
)


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠  No CUDA GPU found – running on CPU (will be very slow).")

    t0 = time.time()

    # ── 1. Load base pipeline ──────────────────
    pipe = load_sdxl_pipeline(args.checkpoint, device)

    # ── 2. Load ControlNet ─────────────────────
    controlnet = load_controlnet(args.controlnet, device)

    # ── 3. Build ControlNet pipeline ──────────
    from diffusers import StableDiffusionXLControlNetPipeline

    print("[4/5] Assembling ControlNet + IPAdapter pipeline …")
    cn_pipe = StableDiffusionXLControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
    )
    cn_pipe.enable_attention_slicing(slice_size="auto")
    cn_pipe.vae.enable_tiling()
    cn_pipe.vae.enable_slicing()
    try:
        cn_pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    cn_pipe.enable_sequential_cpu_offload(gpu_id=0)

    del pipe
    free_vram()

    # ── 4. Prepare conditioning images ────────
    W, H = args.width, args.height

    ref_img   = load_image_as_tensor(args.ref_image,      W, H)   # IPAdapter reference
    depth_img = load_image_as_tensor(args.depth_image,    W, H)   # ControlNet #1 (depth)
    norm_img  = load_image_as_tensor(args.normals_image,  W, H)   # ControlNet #2 (normals)

    # PIL versions for diffusers API
    ref_pil   = tensor_to_pil(ref_img)
    depth_pil = tensor_to_pil(depth_img)
    norm_pil  = tensor_to_pil(norm_img)

    # ── 5. IPAdapter injection ─────────────────
    #
    # If ip-adapter package is available, inject style transfer.
    # Otherwise fall back to image prompt (less precise).
    #
    ip_image_prompt_embeds = None
    ip_uncond_image_prompt_embeds = None

    try:
        from ip_adapter import IPAdapterXL
        from transformers import CLIPVisionModelWithProjection

        clip_vision_dir = str(Path(args.clip_vision).parent)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            clip_vision_dir,
            torch_dtype=torch.float16,
        ).to(device)

        # Load ip-adapter weights manually and extract image embeddings
        import safetensors.torch as sf_torch
        ip_weights = sf_torch.load_file(args.ipadapter)

        # Encode reference image with CLIP vision
        from transformers import CLIPImageProcessor
        clip_processor = CLIPImageProcessor()
        clip_inputs = clip_processor(images=ref_pil, return_tensors="pt").to(device)

        with torch.inference_mode():
            image_features = image_encoder(**clip_inputs).image_embeds  # (1, 1280)

        # Project to UNet cross-attention dimension via ip-adapter proj layers
        # (weight keys: "image_proj.*" and "ip_adapter.*")
        proj_weight = ip_weights.get("image_proj.proj.weight")
        proj_bias   = ip_weights.get("image_proj.proj.bias")
        if proj_weight is not None:
            proj_weight = proj_weight.to(device, dtype=torch.float16)
            proj_bias   = proj_bias.to(device, dtype=torch.float16) if proj_bias is not None else None
            ip_image_prompt_embeds = torch.nn.functional.linear(
                image_features.half(), proj_weight, proj_bias
            ).unsqueeze(1)   # (1, 1, D)
            ip_uncond_image_prompt_embeds = torch.zeros_like(ip_image_prompt_embeds)

        del image_encoder, ip_weights, clip_inputs, image_features
        free_vram()
        print("  ✓ IPAdapter embeddings computed")

    except Exception as e:
        print(f"  ⚠ IPAdapter embedding failed ({e}). Using text-only conditioning.")

    # ── 6. Sampling ───────────────────────────
    print("[5/5] Running KSampler …")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Workflow uses TWO ControlNet passes (depth + normals).
    # diffusers MultiControlNet handles this with a list.
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    # Reload as multi-controlnet if normals != depth
    if args.depth_image != args.normals_image:
        controlnet2 = load_controlnet(args.controlnet, device)
        from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
        multi_cn = MultiControlNetModel([controlnet, controlnet2])

        cn_pipe.controlnet = multi_cn
        control_images  = [depth_pil, norm_pil]
        control_weights = [1.1, 1.0]           # node 95 strength=1.1, node 44 strength=1.0
    else:
        control_images  = [depth_pil]
        control_weights = [1.1]

    # Build call kwargs
    call_kwargs = dict(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_images,
        controlnet_conditioning_scale=control_weights,
        width=W,
        height=H,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        output_type="pil",
    )

    # Inject ip-adapter embeddings if computed
    if ip_image_prompt_embeds is not None:
        call_kwargs["ip_adapter_image_embeds"] = [ip_image_prompt_embeds]

    result = cn_pipe(**call_kwargs)
    image_out: Image.Image = result.images[0]

    # ── 7. Save ───────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_out.save(str(out_path))

    elapsed = time.time() - t0
    print(f"\n✅ Saved → {out_path}  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multiview SDXL generation (ComfyUI workflow → standalone script)"
    )
    # Required paths
    p.add_argument("--checkpoint",    required=True, help=".safetensors SDXL checkpoint")
    p.add_argument("--controlnet",    required=True, help="ControlNet .safetensors (promaxx)")
    p.add_argument("--ipadapter",     required=True, help="ip-adapter_sdxl.safetensors")
    p.add_argument("--clip_vision",   required=True, help="clip_vision_g.safetensors")
    p.add_argument("--ref_image",     required=True, help="Reference image for IPAdapter style transfer")
    p.add_argument("--depth_image",   required=True, help="Depth collage control image")
    p.add_argument("--normals_image", required=True, help="Normals collage control image")
    p.add_argument("--output",        required=True, help="Output image path (.png)")
    # Optional sampler params (mirroring the workflow)
    p.add_argument("--seed",   type=int,   default=871128783595447, help="RNG seed")
    p.add_argument("--steps",  type=int,   default=30,              help="Denoising steps")
    p.add_argument("--cfg",    type=float, default=9.0,             help="Classifier-free guidance scale")
    p.add_argument("--width",  type=int,   default=1024,            help="Output width")
    p.add_argument("--height", type=int,   default=1024,            help="Output height")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate(args)
