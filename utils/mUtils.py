
import maya.cmds as cmds
import os

class mUtils:
    def __init__(self):
        pass

    def extract_maya_fbx(self, fbx_path, output_dir):
        import zipfile
        if not zipfile.is_zipfile(fbx_path):
            print(f"{fbx_path} is not a valid zip file.")
            return False

        with zipfile.ZipFile(fbx_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {fbx_path} to {output_dir}")
        return True