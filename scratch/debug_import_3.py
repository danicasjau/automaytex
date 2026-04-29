import sys, os
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

print("Path adjusted. Attempting full import block...")

try:
    from transformers import (
        CLIPTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        DepthAnythingForDepthEstimation,
        AutoImageProcessor,
    )
    print("Successfully imported all from transformers")
except Exception as e:
    import traceback
    traceback.print_exc()
