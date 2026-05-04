
import sys
import os

from mPipline.geoExtraction import meshCollage_generator, meshReProjectUV, meshTetrahedron_render

## UTILS



class MeshRenderer:
    def __init__(self, config):
        self.conf = config

        print(config)
        print(self.conf)

        self.retargetTool = meshReProjectUV.UVRetargetTool(
            output_dir=self.conf.temporal_path,
            config=self.conf
        )

        self.outputs = None
        
    def renderMesh(self):
        self.retargetTool.getOriginalUV()

        FACE_ORDER = ["face_0", "face_1", "face_2", "face_3"]

        extractor = meshTetrahedron_render.GeometryPlanarExtractor(
            export_path=self.conf.temporal_path,
            resolution=self.conf.resolution/2,
            camera_scale=self.conf.camera_scale
        )

        extractor.run()
        images = [os.path.join(self.conf.temporal_path, f"{face}.exr") for face in FACE_ORDER]
        
        gen = meshCollage_generator.EXRCollageGenerator(
            image_paths      = images,
            save_path        = self.conf.output_path,
            depth_saturation = self.conf.depth_saturation,
            resize_to        = self.conf.resolution,
        )

        self.outputs = gen.run()
        return self.outputs

    def getOutputs(self):
        return self.outputs

def main():
    renderer = MeshRenderer()
    renderer.renderMesh()