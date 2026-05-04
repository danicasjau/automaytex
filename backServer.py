

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

def _load_all_models():
    res = requests.get(f"{BASE_URL}/loadallmodels")
    return res.json()

def _unload_all_models():
    res = requests.get(f"{BASE_URL}/unloadallmodels")
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

