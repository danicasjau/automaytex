#!/usr/bin/env bash
# run_multiview.sh — convenience launcher for multiview_generate.py
# Edit the MODEL_DIR and IMAGE_DIR variables to match your setup.
# Then: chmod +x run_multiview.sh && ./run_multiview.sh

import subprocess
import multiview_generate

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR="D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/models"
IMAGE_DIR="D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/images"
OUTPUT="D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/images/multiview_out.png"

CHECKPOINT="E:/Program Files/ComfyUI/ComfyUI/models/checkpoints/juggernautXL_v9Rdphoto2Lightning.safetensors"
CONTROLNET="E:/Program Files/ComfyUI/ComfyUI/models/controlnet/diffusion_pytorch_model_promaxx.safetensors"
IPADAPTER="D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/models/ip-adapter/ip-adapter_sdxl.safetensors"
CLIP_VISION="D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/models/clip/clip_vision_g.safetensors"

REF_IMAGE=f"{IMAGE_DIR}/reference.png"
DEPTH_IMAGE=f"{IMAGE_DIR}/collage_depth.png"
NORMALS_IMAGE=f"{IMAGE_DIR}/collage_normals.png"


# ── Sampler params (mirror the workflow) ─────────────────────────────────────
SEED=871128783595447
STEPS=30
CFG=9.0
WIDTH=1024
HEIGHT=1024

class defA:
    checkpoint = CHECKPOINT
    controlnet = CONTROLNET
    ipadapter = IPADAPTER
    clip_vision = CLIP_VISION
    ref_image = REF_IMAGE
    depth_image = DEPTH_IMAGE
    normals_image = NORMALS_IMAGE
    output = OUTPUT
    seed = SEED
    steps = STEPS
    cfg = CFG
    width = WIDTH
    height = HEIGHT

multiview_generate.generate(defA())

