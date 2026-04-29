import sys, os
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)

try:
    from transformers.models.clip.modeling_clip import CLIPTextModel
    print("Successfully imported CLIPTextModel from modeling_clip")
except Exception as e:
    import traceback
    traceback.print_exc()
