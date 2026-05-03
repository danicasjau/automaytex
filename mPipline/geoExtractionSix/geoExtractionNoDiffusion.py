import os
import sys

sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex")
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

from config import configuration
from mPipline.geoExtractionSix.geoPlanarExtraction import GeometryPlanarExtractor
from mPipline.exrCollage.exrCollageGenerator import EXRCollageGenerator
from mPipline.exrCollage.exrCollageGenerator import EXRCollageGenerator
from mPipline.geoExtractionSix.geoPlanarUVProjection import GeoPlanarUVProjection
from mPipline.exrCollage.exrCollageBroker import CollageSplitter
from mPipline.mtlMaya.materialCreation import autoMaMaterial
import maya.cmds as cmds

class geoExtractionPipeline:
    def __init__(self):
        self.config = configuration()
        self.selection = cmds.ls(sl=True, long=True)
        
        # 1. Initialize the Maya EXR Extractor
        self.extractor = GeometryPlanarExtractor(
            export_path=self.config.temporal_path,
            resolution=self.config.resolution
        )
        
        # 2. Build the list of expected EXR files 
        self.face_images = [
            os.path.join(self.config.temporal_path, f"{face}.exr") 
            for face in EXRCollageGenerator.FACE_ORDER
        ]
        
        # 3. Initialize the EXR Collage Generator
        self.prep = EXRCollageGenerator(
            image_paths=self.face_images,
            save_path=self.config.output_path,
            depth_saturation=0.5,
            resize_to=self.config.resolution
        )
        
        # 4. UV Projector
        self.uv_projector = GeoPlanarUVProjection()

    def run(self):
        print("====== Starting Geometry Extraction Pipeline ======")
        # 1. Render all EXRs from Maya
        print("\n--- 1. Baking EXRs ---")
        self.extractor.run()
        
        # 2. Project UVs
        print("\n--- 2. Building UVs ---")
        self.uv_projector.run(selection=self.selection)
        
        # 3. Build collages from the EXRs
        print("\n--- 3. Building Collages ---")
        outputs = self.prep.run()
        
        # 4. Split Collage RGBA back into UDIM mapping
        print("\n--- 4. Splitting RGBA Collage ---")
        rgba_files = []
        rgba_path = outputs.get("rgba")
        if rgba_path and os.path.exists(rgba_path):
            rgba_splitter = CollageSplitter(
                collage_path=rgba_path,
                material_name="extracted_rgba",
                output_root=self.config.output_path,
                output_size=self.config.resolution
            )
            rgba_files = rgba_splitter.run()
            
        # 5. Apply Material with UDIMs
        if rgba_files and self.selection:
            print("\n--- 5. Applying Material and Textures ---")
            
            rgba_1001 = next((p for p in rgba_files if "1001" in p), rgba_files[0])
                
            # Create a clean material for extraction results
            mat_creator = autoMaMaterial()
            mat_creator.mName = "PlanarExtracted_MAT"
            mat_creator.mObject = self.selection
            mat_creator.create()
            
            # Connect the RGBA map sequence to diffuse
            mat_creator.connectImage(rgba_1001, slot="diffuse", udim=True)
            
            # Assign the material to selection
            mat_creator.assign_to_object()
        
        print("\n====== Pipeline Complete ======")
        for channel, path in outputs.items():
            print(f"  {channel:8s} -> {path}")
            
        print("\n  Extracted RGBA UDIMs:")
        for sf in rgba_files:
            print(f"  -> {sf}")

g = geoExtractionPipeline()
g.run()