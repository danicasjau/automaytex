###############################################
## IMPORTING LIBRARIES
################################################

import json
import os
import sys
import importlib

################################################
## SETTING UP CONFIGURATION from CONFIGURATION FILE
################################################

_BASE_PATH = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"
_CONFIG_PATH = os.path.join(_BASE_PATH, "data", "configuration.json")

_config_data = {}
try:
    with open(_CONFIG_PATH, "r") as _f:
        _config_data = json.load(_f)
except Exception as e:
    print(f"Error loading configuration.json: {e}")

# Set environment variables for the pipeline
os.environ["BASE_DIR"] = str(_config_data.get("BASE_DIR", ""))
os.environ["ENV_PATH"] = str(_config_data.get("ENV_PATH", ""))
os.environ["SCRIPTS_PATH"] = str(_config_data.get("SCRIPTS_PATH", ""))
os.environ["MODELS_PATH"] = str(_config_data.get("MODELS_PATH", ""))

sys.path.append(_BASE_PATH)
sys.path.append(os.path.join(_config_data.get("ENV_PATH", ""), "lib", "site-packages"))

import config as conf

_general_configuration = conf.configuration()


################################################
## IMPORTING PIPELINES
################################################

import mGui
import autotex

importlib.reload(mGui)
importlib.reload(autotex)

import mGui as mayagui
import autotex

# Keep a global reference to the window so it isn't garbage collected by Maya
_automaytex_window = None

def start_gui():
    """Extracts user settings from the GUI and connects the signal."""
    global _automaytex_window
    _automaytex_window = mayagui.automaytexGUI()
    _automaytex_window.texturize_signal.connect(start_pipeline)
    _automaytex_window.show()

def start_pipeline(_configuration=None):
    print(f"Starting pipeline with following data: {_configuration.printdata()}")
    pipeline = autotex.autoTexturePipeline(_configuration)
    pipeline.run()


if __name__ == "__main__":
    start_gui()