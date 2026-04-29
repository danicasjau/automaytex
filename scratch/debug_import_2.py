import sys, os
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

print("Path adjusted. Attempting import...")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    from transformers.models.clip.modeling_clip import CLIPTextModel
    print("Successfully imported CLIPTextModel")
except Exception as e:
    import traceback
    traceback.print_exc()
    # Try to see if it's a specific requirement failure
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        import numpy
        print(f"Numpy version: {numpy.__version__}")
    except:
        print("Failed to import torch or numpy")
