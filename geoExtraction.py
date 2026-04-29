import os
import sys

sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex")
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

import importlib
import config
import geoPlanarExtraction
import exrCollageGenerator
import geoPlanarUVProjection
import exrCollageBroker
import materialCreation
import reUvPorjection

importlib.reload(config)
importlib.reload(geoPlanarExtraction)
importlib.reload(exrCollageGenerator)
importlib.reload(geoPlanarUVProjection)
importlib.reload(exrCollageBroker)
importlib.reload(materialCreation)
importlib.reload(reUvPorjection)

from config import configuration
from geoPlanarExtraction import GeometryPlanarExtractor
from exrCollageGenerator import EXRCollageGenerator
from geoPlanarUVProjection import GeoPlanarUVProjection
from exrCollageBroker import CollageSplitter
from materialCreation import autoMaMaterial
from reUvPorjection import UVRetargetTool

import maya.cmds as cmds
import glob    
import subprocess


class geoExtractionPipeline:

    def __init__(self):

        self.config = configuration()
        self.selection = cmds.ls(sl=True, long=True)

        self.renderImages = True    
        self.diffuseGeneration = True
        self.upscaleTextures = True
        self.fixSeams = True

        # 1. Initialize the Maya EXR Extractor
        self.extractor = GeometryPlanarExtractor(
            export_path=self.config.temporal_path,
            resolution=self.config.resolution
        )
        
        # 2. Build the list of expected EXR files 
        self.face_images = [
            os.path.join(self.config.temporal_path, f"{face}.exr") 
            for face in self.config.face_order
        ]
        
        # 3. Initialize the EXR Collage Generator
        self.prep = EXRCollageGenerator(
            image_paths=self.face_images,
            save_path=self.config.output_path,
            depth_saturation=self.config.depth_saturation,
            resize_to=self.config.resolution
        )
        
        # 4. UV Projector
        self.uv_projector = GeoPlanarUVProjection(config=self.config)

    def run(self):

        # 0. Initialise Retarget Tool
        print("\n--- 0. Retarget Tool Linked ---")
        self.retarget_tool = UVRetargetTool(self.selection[0], output_dir=self.config.output_path, config=self.config)
        self.retarget_tool.getOriginalUV() # Capture original UVs before projection

        print("\n--- 1. Baking EXRs ---")
        if self.renderImages:
            self.extractor.run()
        
        print("\n--- 2. Building UVs ---")
        self.uv_projector.run(selection=self.selection)
        
        print("\n--- 3. Building Collages ---")
        outputs = self.prep.run()
        

        depth_path = outputs.get("depth")
        normals_path = outputs.get("normals")
        gen_diffuse_path = None

        if depth_path and os.path.exists(depth_path) and self.diffuseGeneration:
            print("\n--- 4. AI Image2Image Generation ---")
            
            gen_diffuse_path = os.path.join(self.config.output_path, "collage_diffuse.png")
            python_exe = self.config.python_exe
            script_path = os.path.join(self.config.base_dir, self.config.script_name)

            try:
                subprocess.run([
                    python_exe, script_path, 
                    "--prompt", self.config.prompt, 
                    "--depth", depth_path, 
                    "--output", gen_diffuse_path
                ], check=True)

            except subprocess.CalledProcessError as e:
                print(f"Diffusion Generation Failed: {e}")
                gen_diffuse_path = None
        else:
            print("[Warning] Skipping AI Generation or Input Missing. Falling back to normals.")
            gen_diffuse_path = os.path.join(self.config.output_path, "collage_normals.png")


        print("\n--- 5. Splitting Diffuse Collage ---")

        diffuse_files = []
        if gen_diffuse_path and os.path.exists(gen_diffuse_path):

            diffuse_splitter = CollageSplitter(
                collage_path=gen_diffuse_path,
                material_name=self.config.material_base_name,
                output_root=self.config.output_path,
                output_size=self.config.resolution
            )
            diffuse_files = diffuse_splitter.run()
            
        # 6. Apply Planar Extracted Material (Intermediate)
        # We only do this if we don't have a retarget tool or explicitly want intermediate
        if diffuse_files and self.selection and not getattr(self, 'do_retarget', True):
            print("\n--- 6. Applying Material and Textures (Planar) ---")
            diffuse_1001 = next((p for p in diffuse_files if "1001" in p), diffuse_files[0])
            mat_creator = autoMaMaterial(config=self.config)
            mat_creator.mName = "AI_Wooden_Planar_MAT"
            mat_creator.mObject = self.selection
            mat_creator.create()
            mat_creator.connectImage(diffuse_1001, slot="diffuse", udim=True)
            mat_creator.assign_to_object()
        
        # 7. Retarget and Apply Final Material
        if self.retarget_tool and diffuse_files:
            print("\n--- 7. Retargeting Texture to Original UVs ---")
            self.retarget_tool.setMaterialTextures(diffuse_files)
            self.retarget_tool.retargetToOriginalUV(resolution=self.config.resolution)
            
            # Look specifically for the files we just generated
            retargeted_files = glob.glob(os.path.join(self.config.output_path, "retarget_*.png"))
            
            # 7.5 AI Seam Fixing (Retargeted)
            if retargeted_files and self.fixSeams:
                print("\n--- 7.5. AI Seam Fixing (Retargeted Tiles) ---")
                fixed_retargeted_files = []
                python_exe = self.config.python_exe
                solver_script = os.path.join(self.config.base_dir, self.config.seam_fixer_script)
                
                for retarget_path in retargeted_files:
                    base_name = os.path.splitext(os.path.basename(retarget_path))[0]
                    fixed_path = os.path.join(self.config.output_path, f"{base_name}_fixed.png")
                    
                    print(f"[Info] Fixing seams for {os.path.basename(retarget_path)}...")
                    try:
                        subprocess.run([
                            python_exe, solver_script,
                            "--input", retarget_path,
                            "--output", fixed_path,
                            "--prompt", self.config.prompt,
                            "--strength", str(self.config.seam_fixer_strength),
                            "--steps", str(self.config.seam_fixer_steps)
                        ], check=True)
                        
                        if os.path.exists(fixed_path):
                            fixed_retargeted_files.append(fixed_path)
                            print(f"  -> Fixed: {os.path.basename(fixed_path)}")
                        else:
                            fixed_retargeted_files.append(retarget_path)
                    except subprocess.CalledProcessError as e:
                        print(f"[Error] Seam Fixing Failed for {retarget_path}: {e}")
                        fixed_retargeted_files.append(retarget_path)
                
                retargeted_files = fixed_retargeted_files
            
            if retargeted_files:
                # 8. AI Upscaling / Polishing
                if self.upscaleTextures:
                    print("\n--- 8. AI Upscaling / Polishing ---")
                    
                    python_exe = self.config.python_exe
                    upscaler_script = os.path.join(self.config.base_dir, "upscalerPolishing.py")
                    
                    try:
                        # Build argument list: --input path1 path2 ...
                        cmd = [python_exe, upscaler_script, "--output", self.config.output_path, "--res", "2k", "--input"]
                        cmd.extend(retargeted_files)
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"\n[Error] Upscaling subprocess failed (Exit {result.returncode})")
                            print(f"[Stdout] {result.stdout}")
                            print(f"[Stderr] {result.stderr}")
                            raise subprocess.CalledProcessError(result.returncode, cmd)
                        
                        # The upscaler names files: {orig_name}_upscaled.png
                        upscaled_files = []
                        for original_path in retargeted_files:
                            base_name = os.path.splitext(os.path.basename(original_path))[0]
                            upscaled_path = os.path.join(self.config.output_path, f"{base_name}_upscaled.png")
                            if os.path.exists(upscaled_path):
                                upscaled_files.append(upscaled_path)
                        
                        if upscaled_files:
                            retargeted_files = upscaled_files
                            print(f"[Info] Successfully upscaled {len(retargeted_files)} tiles via subprocess.")
                            
                    except Exception as e:
                        print(f"[Warning] Upscaling subprocess encountered an error: {e}")

                print("\n--- 9. Applying Final Material ---")
                final_1001 = next((p for p in retargeted_files if "1001" in p), retargeted_files[0])
                
                mat_creator_retarget = autoMaMaterial(config=self.config)
                mat_creator_retarget.mName = "AI_Wooden_Final_MAT"
                mat_creator_retarget.mObject = self.selection
                mat_creator_retarget.create()
                mat_creator_retarget.connectImage(final_1001, slot="diffuse", udim=True)
                mat_creator_retarget.assign_to_object()
                
                diffuse_files = retargeted_files


                    
        print("\n====== Pipeline Complete ======")
        for channel, path in outputs.items():
            print(f"  {channel:8s} -> {path}")
            
        print("\n  Extracted Diffuse UDIMs:")
        for sf in diffuse_files:
            print(f"  -> {sf}")

g = geoExtractionPipeline()
g.run()