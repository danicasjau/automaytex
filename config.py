from dataclasses import dataclass

@dataclass
class validation:
    base_model = ["sdxl", "sd15"]

    quantization = [None, "fp16", "int8", "int4", "bf16", "fp32"]

    material_types = ["mtlx", "standard"]

    texture_resolutions = {
        "512": 512,
        "1k": 1024,
        "2k": 2048,
        "4k": 4096,
        "8k": 8194
    }


@dataclass
class configuration:

    BASE_DIR = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"

    python_exe = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Scripts\python.exe"
    script_name = "mPiplineCreationSDXL.py"

    # MODELS PATHS

    base_model = "sdxl"

    controlnet_model = "diffusers/controlnet-normal-sdxl-1.0"

    ip_adapter_model = "h94/IP-Adapter"
    ip_adapter_subfolder = "sdxl_models"
    ip_adapter_weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_adapter_scale = 0.7

    quantization = "fp16"

    # PATHS
 
    material_name = f"material01"

    temporal_path = f"{BASE_DIR}/output/{material_name}/temp"
    textures_path = f"{BASE_DIR}/output/{material_name}/textures"

    output_path = f"{BASE_DIR}/output/{material_name}"

    # GENERATION
    # -- gui editable variables start --

    positive_prompt = """
    8K ultra-detailed seamless texture of white wood paneling, no seams, tileable, photorealistic
    """

    negative_prompt = """
    blurry, low quality, distorted, shadow, lighting gradients
    """

    texture_resolution = "1k"

    inference_steps = 16
    cfg_scale = 7.5

    noise = 0.05
    seed = 123456789

    generated_images = ["diffuse", "roughness", "metalness", "normal", "height"]

    # -------------------------------
    # SYSTEM 
    # -------------------------------

    system_prfered = "gpu"

    # -------------------------------
    # MAYA MATERIAL ASSIGNMENT 
    # -------------------------------

    assign_maya_material = True
    material_type = "mtlx"


    # -------------------------------
    # EXTRA SETTINGS 
    # -------------------------------
    
    uv_chunk_size = 500
    ortho_padding = 0.08
    depth_saturation = 0.5

    material_base_name = "extracted_diffuse"
    retarget_uv_set_name = "retargetUV"
    camera_name = "planarExtractCam"
    
    seam_fixer_script = "mPiplineDiffsuionSolver.py"
    seam_fixer_strength = 0.55
    seam_fixer_steps = 25

    def printdata(self):
        return f"""
        BASE MODEL: {self.base_model}
        CONTROLNET MODEL: {self.controlnet_model}
        IP-ADAPTER MODEL: {self.ip_adapter_model}
        IP-ADAPTER SCALE: {self.ip_adapter_scale}
        QUANTIZATION: {self.quantization}
        MATERIAL TYPE: {self.material_type}
        TEXTURE RESOLUTION: {self.texture_resolution}
        POSITIVE PROMPT: {self.positive_prompt}
        NEGATIVE PROMPT: {self.negative_prompt}
        INFERENCE STEPS: {self.inference_steps}
        CFG SCALE: {self.cfg_scale}
        NOISE: {self.noise}
        SEED: {self.seed}
        GENERATsED IMAGES: {self.generated_images}
        SYSTEM PREFERENCE: {self.system_prfered}
        ASSIGN MAYA MATERIAL: {self.assign_maya_material}
        UV CHUNK SIZE: {self.uv_chunk_size}
        ORTHO PADDING: {self.ortho_padding}
        DEPTH SATURATION: {self.depth_saturation}
        MATERIAL BASE NAME: {self.material_base_name}
        RETARGET UV SET NAME: {self.retarget_uv_set_name}
        CAMERA NAME: {self.camera_name}
        SEAM FIXER STRENGTH: {self.seam_fixer_strength}
        SEAM FIXER STEPS: {self.seam_fixer_steps}
        """

    def validate(self):
        v = validation()
        if self.material_type not in v.material_types:
            raise ValueError(f"Invalid material type: {self.material_type}. Must be one of {v.material_types}.")
        if self.texture_resolution not in v.texture_resolutions:
            raise ValueError(f"Invalid texture resolution: {self.texture_resolution}. Must be one of {list(v.texture_resolutions.keys())}.")
        if self.base_model not in v.base_model:
            raise ValueError(f"Invalid base model: {self.base_model}. Must be one of {v.base_model}.")
        


if __name__ == "__main__":
    config = configuration()
    config.base_model = "sd154334"
    config.validate()
    print(config.printdata())