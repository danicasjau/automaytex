
import sys
import os
import importlib


# Add local paths
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex")
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\regularTetrahedronExtraction")

sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages")

import meshCollage_generator
import meshReProjectUV
import meshTetrahedron_render
import config

importlib.reload(meshReProjectUV)
importlib.reload(meshTetrahedron_render)
importlib.reload(meshCollage_generator)
importlib.reload(config)

import meshCollage_generator as mg
import meshTetrahedron_render as mr
import meshReProjectUV as rp

from config import configuration


class MeshRenderer:
    def __init__(self):
        self.conf = configuration()

        self.retargetTool = rp.UVRetargetTool(
            output_dir=self.conf.temporal_path,
            config=self.conf
        )
        
    def renderMesh(self):
        self.retargetTool.getOriginalUV()

        conf = configuration()

        FACE_ORDER = ["face_0", "face_1", "face_2", "face_3"]

        extractor = mr.GeometryPlanarExtractor(
            export_path=conf.temporal_path,
            resolution=conf.resolution/2,
        )
        extractor.run()
        
        images = [os.path.join(conf.temporal_path, f"{face}.exr") for face in FACE_ORDER]
        
        gen = mg.EXRCollageGenerator(
            image_paths      = images,
            save_path        = conf.output_path,
            depth_saturation = conf.depth_saturation,
            resize_to        = conf.resolution,
        )

        outputs = gen.run()

        return outputs



def main():
    renderer = MeshRenderer()
    renderer.renderMesh()
    renderer.retargetTexture2UV()



main()