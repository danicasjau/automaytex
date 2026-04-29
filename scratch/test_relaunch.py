import sys, os
_VENV_EXE = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Scripts\python.exe"

print(f"Current Python: {sys.executable}")

if sys.executable.lower() != _VENV_EXE.lower() and os.path.exists(_VENV_EXE):
    print(f"Relaunching with: {_VENV_EXE}")
    os.execv(_VENV_EXE, [_VENV_EXE] + sys.argv)

print("Running in the correct environment now!")
import torch
print(f"Torch loaded from: {torch.__file__}")
print(f"Torch version: {torch.__version__}")
