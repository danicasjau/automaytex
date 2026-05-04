
import sys
import os

from mPipline.geoExtractionSix import geoPlanarExtraction, geoPlanarReProjectUV
from mPipline.exrCollage.exrCollageGenerator import EXRCollageGenerator

class SixMeshRenderer:
    def __init__(self, config):
        self.conf = config

        self.retargetTool = geoPlanarReProjectUV.GeoPlanarUVProjection(
            output_dir=self.conf.temporal_path,
            config=self.conf
        )

        self.outputs = None
        
    def renderMesh(self):
        extractor = geoPlanarExtraction.GeometryPlanarExtractor(
            export_path=self.conf.temporal_path,
            resolution=self.conf.resolution/2,
            config=self.conf
        )

        extractor.run()
        images = [os.path.join(self.conf.temporal_path, f"{face}.exr") for face in self.conf.face_order_6]
        
        gen = EXRCollageGenerator(
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
    renderer = SixMeshRenderer()
    renderer.renderMesh()