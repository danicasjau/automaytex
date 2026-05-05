

import subprocess
import sys
import time
import requests
import os

PORT = 8001
HOST = "127.0.0.1"
ENVIRONMENT_PATH = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Scripts\python.exe"
BASE_URL = f"http://{HOST}:{PORT}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CREATE_NEW_CONSOLE = 0x00000010

def _load_all_models(config=None):
    print("\n\n\nSending payload: \n", config)
    payload = config.dict()
    res = requests.post(f"{BASE_URL}/loadallmodels", json=payload)
    return res.json()

def _unload_all_models():
    res = requests.get(f"{BASE_URL}/unloadallmodels")
    return res.json()

def _generate_texture(configuration, image_path):
    payload = configuration.dict()
    
    # Add mapped/missing parameters for DiffGenSDXL
    payload["input_image"] = image_path
    payload["steps"] = payload.get("inference_steps", 30)
    payload["cfg"] = payload.get("cfg_scale", 8.0)
    payload["controlnet_strength"] = 1.0  # Default or pull from config if added
    payload["scale"] = 1.0                # Default
    payload["depth_resolution"] = 1024    # Default

    print("\n\n\nSending payload: \n", payload)
    res = requests.post(f"{BASE_URL}/generatetexture", json=payload)
    return res.json()

def start_server():
    print("Starting server...")
    # Use Popen so it doesn't block Maya's main thread
    return subprocess.Popen(
        [
            ENVIRONMENT_PATH, "-m", "uvicorn",
            "server.server:app",
            "--host", HOST,
            "--port", str(PORT)
        ],
        creationflags=CREATE_NEW_CONSOLE,
        cwd=BASE_DIR
    )

def stop_server(proc):
    proc.terminate()

