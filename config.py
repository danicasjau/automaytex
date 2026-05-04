# automaytex\config.py
# CONFIGURATION .py file

from dataclasses import dataclass
import os

################################################
## VALIDATION VARIABLES
################################################

@dataclass
class validation:
    base_model = ["sdxl", "fast_sdxl", "flash_sdxl"]
    quantization = [None, "fp16", "int8", "int4", "bf16", "fp32"]
    material_types = ["mtlx", "standard"]

    texture_resolutions = {
        "512": 512,
        "1k": 1024,
        "2k": 2048,
        "4k": 4096,
        "8k": 8194
    }


################################################
## DEFAULT PATHS - from JSON CONFIG FILE
################################################

class paths:
    BASE_DIR = os.environ.get("BASE_DIR")
    ENV_PATH = os.environ.get("ENV_PATH")
    SCRIPTS_PATH = os.environ.get("SCRIPTS_PATH")
    MODELS_PATH = os.environ.get("MODELS_PATH")

    python_exe = os.path.join(ENV_PATH or "", "Scripts", "python.exe")

    # MODELS PATHSs
    diffusion_model = os.path.join(MODELS_PATH or "", "checkpoints", "juggernautXL_v9Rdphoto2Lightning.safetensors")
    controlnet_model = os.path.join(MODELS_PATH or "", "controlnet", "diffusion_pytorch_model_promaxx.safetensors")
    depth_model = os.path.join(MODELS_PATH or "", "models", "depth_anything_vitl14")





################################################
## DEFAULT CONFIGURATION
################################################

@dataclass
class configuration:
    
    # System Paths
    material_name = f"basicmaterial"

    renderMode = "cube"
    camera_name = f"_temp_camera"

    SERVER = False

    script_name = "mPiplineCreationSDXL.py"
    models_json = r"D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/data/models.json"

    # MODELS PATHS

    base_model = "fast_sdxl"
    controlnet_model = "diffusers/controlnet-normal-sdxl-1.0"

    ip_adapter_model = "h94/IP-Adapter"
    ip_adapter_subfolder = "sdxl_models"
    ip_adapter_weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_adapter_scale = 0.7


    quantization = "fp16"
    resolution = 1024

    # PATHS
    temporal_path = f"{paths.BASE_DIR}/output/{material_name}/temp"
    textures_path = f"{paths.BASE_DIR}/output/{material_name}/textures"
    output_path = f"{paths.BASE_DIR}/output/{material_name}"

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

    face_order = ["face_0", "face_1", "face_2", "face_3"]

    face_order_6 = ["top", "bottom", "front", "back", "right", "left"]

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
    camera_scale = 1

    
    depth_saturation = 0.5

    material_base_name = "extracted_diffuse"
    retarget_uv_set_name = "retargetUV"
    camera_name = "planarExtractCam"
    
    seam_fixer_script = "mPiplineDiffsuionSolver.py"
    seam_fixer_strength = 0.55
    seam_fixer_steps = 25

    retargetUV = True

    view_rotations = {
    "top":    (-90,  0,  0),
    "bottom": ( 90,  0,  0),
    "front":  ( 0,  0,  0),
    "back":   ( 0,180,  0),
    "left":   ( 0, 90,  0),
    "right":  ( 0,-90,  0),
    }

    def printdata(self):
        return f"""
        MATERIAL NAME: {self.material_name}
        OUTPUT PATH: {self.output_path}



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

    def dict(self):
        return {
            "material_name": self.material_name,
            "output_path": self.output_path,

            "base_model": self.base_model,
            "controlnet_model": self.controlnet_model,
            "ip_adapter_model": self.ip_adapter_model,
            "ip_adapter_scale": self.ip_adapter_scale,

            "quantization": self.quantization,
            "material_type": self.material_type,
            "texture_resolution": self.texture_resolution,

            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,

            "inference_steps": self.inference_steps,
            "cfg_scale": self.cfg_scale,
            "noise": self.noise,
            "seed": self.seed,
            "generated_images": self.generated_images,
            "system_prfered": self.system_prfered,
            "assign_maya_material": self.assign_maya_material,
            "uv_chunk_size": self.uv_chunk_size,
            "ortho_padding": self.ortho_padding,
            "depth_saturation": self.depth_saturation,
            "material_base_name": self.material_base_name,
            "retarget_uv_set_name": self.retarget_uv_set_name,
            "camera_name": self.camera_name,
            "seam_fixer_strength": self.seam_fixer_strength,
            "seam_fixer_steps": self.seam_fixer_steps,
        }

    def pathsetter(self):
        self.output_path = os.path.join(paths.BASE_DIR, "output", self.material_name)
        self.temporal_path = os.path.join(self.output_path, "temp")
        self.textures_path = os.path.join(self.output_path, "textures")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.temporal_path):
            os.makedirs(self.temporal_path)
        if not os.path.exists(self.textures_path):
            os.makedirs(self.textures_path)

    def validate(self):
        v = validation()
        if self.material_type not in v.material_types:
            raise ValueError(f"Invalid material type: {self.material_type}. Must be one of {v.material_types}.")
        if self.texture_resolution not in v.texture_resolutions:
            raise ValueError(f"Invalid texture resolution: {self.texture_resolution}. Must be one of {list(v.texture_resolutions.keys())}.")
        if self.base_model not in v.base_model:
            raise ValueError(f"Invalid base model: {self.base_model}. Must be one of {v.base_model}.")


