# auto texturing maya - pipline integration
# daniel casadevall jauhiainen

################################################
## IMPORTING LIBRARIES
################################################

import json
import os
import sys
import re
import glob
import subprocess
import importlib

################################################
## IMPORTING MAYA CMDS
################################################

try:
    import maya.cmds as cmds            # type: ignore
    import maya.mel as mel              # type: ignore
    import maya.OpenMaya as om          # type: ignore
except ImportError:
    print("""
    [Warning] Maya cmds module not found. This script should be run inside Maya's Python environment.
    Exiting...
    """)

################################################
## SETTING UP CONFIGURATION from CONFIGURATION FILE
################################################

_BASE_PATH = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"
_CONFIG_PATH = os.path.join(_BASE_PATH, "data", "configuration.json")

_config_data = {}
try:
    with open(_CONFIG_PATH, "r") as _f:
        _config_data = json.load(_f)
except Exception as e:
    print(f"Error loading configuration.json: {e}")

# Set environment variables for the pipeline
os.environ["BASE_DIR"] = str(_config_data.get("BASE_DIR", ""))
os.environ["ENV_PATH"] = str(_config_data.get("ENV_PATH", ""))
os.environ["SCRIPTS_PATH"] = str(_config_data.get("SCRIPTS_PATH", ""))
os.environ["MODELS_PATH"] = str(_config_data.get("MODELS_PATH", ""))

sys.path.append(_BASE_PATH)
sys.path.append(os.path.join(_config_data.get("ENV_PATH", ""), "lib", "site-packages"))

import config as conf

_general_configuration = conf.configuration()


################################################
## IMPORTING PIPELINES
################################################

import mPipline.geoExtraction.geometryRenderer

import mPipline.exrCollage.exrCollageGenerator
import mPipline.exrCollage.exrCollageBroker

import mPipline.mtlMaya.materialCreation
import mPipline.uvUtils.reUvPorjection
import mPipline.mtlMaya.mtlMaterialMapsCreation
import mGui
import cModels

################################################
## RELOADING PIPELINES FOR DEVELOPING
################################################

def maya_remiport_libs():
    importlib.reload(conf)
    
    importlib.reload(mPipline.geoExtraction.geometryRenderer)

    importlib.reload(mPipline.exrCollage.exrCollageGenerator)
    importlib.reload(mPipline.exrCollage.exrCollageBroker)
    
    importlib.reload(mPipline.mtlMaya.materialCreation)
    importlib.reload(mPipline.uvUtils.reUvPorjection)
    importlib.reload(mPipline.mtlMaya.mtlMaterialMapsCreation)
    importlib.reload(mGui)
    importlib.reload(cModels)


################################################
## IMPORTING PIPELINES CLASSES
################################################

from mPipline.geoExtraction.geometryRenderer import MeshRenderer

from mPipline.exrCollage.exrCollageGenerator import EXRCollageGenerator
from mPipline.exrCollage.exrCollageBroker import CollageSplitter

from mPipline.mtlMaya.materialCreation import autoMaMaterial
from mPipline.uvUtils.reUvPorjection import UVRetargetTool
from mPipline.mtlMaya.mtlMaterialMapsCreation import mapsMaterialGenerator

import mGui as mayagui
import cModels


################################################
## MAIN PIPELINE CLASS
################################################

class autoTexturePipeline:
    def __init__(self, _configuration=None):
        if _configuration is None:
            _configuration = _general_configuration
            
        self.config = _configuration

        # maya selection
        self.selection = cmds.ls(sl=True, long=True)

        ################################
        ## PIPELINE DEBUG FLAGS
        ################################
        
        self.debug_renderImages = True    
        self.debug_diffuseGeneration = True
        self.debug_upscaleTextures = True
        self.debug_fixSeams = True

        self.set_tools()


    def set_tools(self):
        print("\n--- Setting tools ---")

        ################################
        ## INITIALIZING PIPELINE OBJECTS
        ################################

        self.extractor = MeshRenderer(
            config=self.config
        )
        
        """
        self.face_images = [
            os.path.join(self.config.temporal_path, f"{face}.exr") 
            for face in self.config.face_order
        ]
        
        self.prep = EXRCollageGenerator(
            image_paths=self.face_images,
            save_path=self.config.output_path,
            depth_saturation=self.config.depth_saturation,
            resize_to=self.config.resolution
        )
        
        # self.uv_projector = GeoPlanarUVProjection(
        #     config=self.config
        # )
        
        self.retarget_tool = UVRetargetTool(
            self.selection[0], 
            output_dir=self.config.output_path, 
            config=self.config
        )"""

    # ===== 2. Helper Methods =====

    def setup_uvs(self):
        print("\n--- 2. Building UVs ---")
        # self.uv_projector.run(selection=self.selection)
        pass

    def create_collages(self):
        print("\n--- 3. Building Collages ---")
        return self.prep.run()

    def retarget_normal(self, normal_collage_path):
        print("\n--- 4. Retargeting Normal to Original UVs ---")

        splitter = CollageSplitter(
            collage_path=normal_collage_path,
            material_name="normal_source",
            output_root=self.config.output_path,
            output_size=self.config.resolution
        )
        normal_udim_tiles = splitter.run()

        self.retarget_tool.setMaterialTextures(normal_udim_tiles)
        self.retarget_tool.retargetToOriginalUV(resolution=self.config.resolution)

        normal_tiles = []
        raw_tiles = glob.glob(os.path.join(self.config.output_path, "retarget_*.png"))
        
        for f in raw_tiles:
            udim_match = re.search(r"10\d{2}", os.path.basename(f))
            if udim_match:
                udim = udim_match.group()
                new_path = os.path.join(self.config.output_path, f"depth_retarget_{udim}.png")
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(f, new_path)
                normal_tiles.append(new_path)
        
        print(f"[Info] Generated {len(normal_tiles)} retargeted normal tiles.")
        return normal_tiles

    def generate_diffuse_textures(self, normal_tiles):
        print("\n--- 5. AI Diffusion Generation (per UDIM) ---")
        if not self.diffuseGeneration or not normal_tiles:
            return []

        print("normal tiles", normal_tiles)

        diffuse_files = []
        python_exe = self.config.python_exe
        script_path = os.path.join(self.config.base_dir, self.config.script_name)

        for depth_tile in normal_tiles:
            udim_match = re.search(r"10\d{2}", os.path.basename(depth_tile))
            udim = udim_match.group() if udim_match else "1001"
            output_path = os.path.join(self.config.output_path, f"diffuse_{udim}.png")

            print(f"[Info] Generating diffuse for UDIM {udim}...")
            try:
                subprocess.run([
                    python_exe, script_path,
                    depth_tile,
                    "--positive-prompt", self.config.prompt,
                    "--negative-prompt", self.config.negative_prompt,
                    "--output", output_path,
                    "--steps", str(self.config.num_inference_steps),
                    "--cfg", str(self.config.guidance_scale),
                    "--seed", str(self.config.seed)
                ], check=True)
                
                if os.path.exists(output_path):
                    diffuse_files.append(output_path)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Diffusion failed for UDIM {udim}: {e}")

        return diffuse_files

    def apply_seam_fixing(self, diffuse_files):
        print("\n--- 6. AI Seam Fixing ---")
        if not self.fixSeams or not diffuse_files:
            return diffuse_files

        fixed_files = []
        python_exe = self.config.python_exe
        solver_script = os.path.join(self.config.base_dir, self.config.seam_fixer_script)

        for df in diffuse_files:
            base_name = os.path.splitext(os.path.basename(df))[0]
            fixed_path = os.path.join(self.config.output_path, f"{base_name}_fixed.png")

            print(f"[Info] Fixing seams for {os.path.basename(df)}...")
            try:
                subprocess.run([
                    python_exe, solver_script,
                    "--input", df,
                    "--output", fixed_path,
                    "--prompt", self.config.prompt,
                    "--strength", str(self.config.seam_fixer_strength),
                    "--steps", str(self.config.seam_fixer_steps)
                ], check=True)
                fixed_files.append(fixed_path if os.path.exists(fixed_path) else df)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Seam fixing failed for {df}: {e}")
                fixed_files.append(df)

        return fixed_files

    def upscale_textures(self, tiles):
        """Step 7: Increase resolution and polish details."""
        print("\n--- 7. AI Upscaling ---")
        if not self.upscaleTextures or not tiles:
            return tiles

        python_exe = self.config.python_exe
        upscaler_script = os.path.join(self.config.base_dir, "upscalerPolishing.py")

        try:
            cmd = [python_exe, upscaler_script, "--output", self.config.output_path, "--res", "2k", "--input"]
            cmd.extend(tiles)
            subprocess.run(cmd, check=True)

            upscaled = []
            for t in tiles:
                base = os.path.splitext(os.path.basename(t))[0]
                up_path = os.path.join(self.config.output_path, f"{base}_upscaled.png")
                upscaled.append(up_path if os.path.exists(up_path) else t)
            return upscaled
        except Exception as e:
            print(f"[Warning] Upscaling failed: {e}")
            return tiles

    def assign_material(self, image_dict):
        """Step 8: Create and assign the final material to the Maya object."""
        print("\n--- 8. Applying Final Material ---")
        if not image_dict or not self.selection:
            return

        mat_creator = autoMaMaterial(config=self.config)
        mat_creator.mName = "AI_Wooden_Final_MAT"
        mat_creator.mObject = self.selection

        mat_creator.create()
        mat_creator.connectImages(image_dict, udim=True)
        mat_creator.assign_to_object()

    def run(self):
        ##########################################
        ## PIPELINE EXECUTION
        ##########################################

        print("\n====== Starting Automay text texturing Expo Pipeline ======")

        # ===== 1. Validation =====
        if not self.selection:
            print("[Error] No object selected. Please select a mesh in Maya.")
            return
        
        # getting original UVs for retargeting later
        # self.retarget_tool.getOriginalUV()

        print("\n --- 1. Baking EXRs ---")

        if self.debug_renderImages:
            self.extractor.renderMesh()

        


        # setting up planar UVs for the extraction

        """
        self.setup_uvs()

        outputs = self.create_collages()
        normal_path = outputs.get("normals")
        print(outputs)

        print("normal path", normal_path)
        
        normal_tiles = self.retarget_normal(normal_path)

        diffuse_files = self.generate_diffuse_textures(normal_tiles)

        print("\nGENERATION COMPLEATED: diffuse_files:", diffuse_files)

        # fixed_files = self.apply_seam_fixing(diffuse_files)

        upscaled_files = self.upscale_textures(diffuse_files)

        mtl = mapsMaterialGenerator(upscaled_files[0], normal_tiles[0])
        mtl.setOutputPath(self.config.output_path)
        mtl.create()


        self.assign_material(mtl.getFiles())
       
        print("\n====== Pipeline Complete ======")
        print(f"Final UDIM Tiles: {len(diffuse_files)}")

        for f in diffuse_files:
            print(f"  -> {os.path.basename(f)}") """



# ============================================
# AUTO PIPELINE EXECUTION
# ============================================

def start_gui():
    """Extracts user settings from the GUI."""
    window = mayagui.automaytexGUI()
    window.set_callback(start_pipeline)
    window.show()

def start_pipeline(_configuration=None):
    print(f"Starting configuration: with following data: {_configuration}")
    # pipeline = autoTexturePipeline(_configuration)
    # pipeline.run()


if __name__ == "__main__":
    start_gui()