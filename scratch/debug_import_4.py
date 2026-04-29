import sys, os
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

print("Path adjusted. Attempting to touch classes...")

try:
    from transformers import (
        CLIPTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        DepthAnythingForDepthEstimation,
        AutoImageProcessor,
    )
    print(f"CLIPTokenizer: {CLIPTokenizer}")
    print(f"CLIPTextModel: {CLIPTextModel}")
    print(f"CLIPTextModelWithProjection: {CLIPTextModelWithProjection}")
    print(f"DepthAnythingForDepthEstimation: {DepthAnythingForDepthEstimation}")
    print(f"AutoImageProcessor: {AutoImageProcessor}")
except Exception as e:
    import traceback
    traceback.print_exc()
