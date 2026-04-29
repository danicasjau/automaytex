from dataclasses import dataclass
@dataclass
class configuration:
    base_dir = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"

    base_model = "models/juggernautXL_ragnarokBy.safetensors"
    controlnet_model = "diffusers/controlnet-normal-sdxl-1.0"

    ip_adapter_model = "h94/IP-Adapter"
    ip_adapter_subfolder = "sdxl_models"
    ip_adapter_weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_adapter_scale = 0.7

    resolution = 1024
    num_inference_steps = 30
    guidance_scale = 7.5
    seed = 42353425345

 
    temporal_path = f"{base_dir}/output2/temp"
    output_path = f"{base_dir}/output2"

    prompt = "8K ultra-detailed seamless texture of white wood paneling, no seams, tileable, photorealistic"
    negative_prompt = "blurry, low quality, distorted, shadow, lighting gradients"

    face_order = ["front", "left", "top", "back", "right", "bottom"]
        
        
    python_exe = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Scripts\python.exe"
    script_name = "mPiplineCreationSDXL.py"

    view_rotations = {
            "front":  ( 0,    0, 0),
            "back":   ( 0,  180, 0),
            "left":   ( 0,   90, 0),
            "right":  ( 0,  -90, 0),
            "top":    (-90,   0, 0),
            "bottom": ( 90,   0, 0),
        }

    uv_chunk_size = 500
    ortho_padding = 0.08
    depth_saturation = 0.5
    material_base_name = "extracted_diffuse"
    retarget_uv_set_name = "retargetUV"
    camera_name = "planarExtractCam"
    
    seam_fixer_script = "mPiplineDiffsuionSolver.py"
    seam_fixer_strength = 0.55
    seam_fixer_steps = 25
