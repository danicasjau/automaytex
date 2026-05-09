

import subprocess
import sys
import time
import requests
import os

PORT = 8001
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_PROJECTS = os.path.dirname(BASE_DIR)
ENVIRONMENT_PATH = os.path.join(BASE_DIR_PROJECTS, "mEnv", "Scripts", "python.exe")

CREATE_NEW_CONSOLE = 0x00000010

import concurrent.futures

def _non_blocking_post(url, json_payload):
    """Executes a POST request in a background thread while keeping the GUI responsive."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        try:
            from PySide2.QtWidgets import QApplication
        except ImportError:
            QApplication = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(requests.post, url, json=json_payload)
        while not future.done():
            if QApplication and QApplication.instance():
                QApplication.processEvents()
            time.sleep(0.05)
        return future.result()

def _load_all_models(config=None):
    print("\n\n\nSending payload: \n", config)
    payload = config.dict()
    res = _non_blocking_post(f"{BASE_URL}/loadallmodels", payload)
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
    res = _non_blocking_post(f"{BASE_URL}/generatetexture", payload)
    return res.json()

def are_models_loaded():
    print("[SERVER] Checking if models are loaded...")
    res = requests.get(f"{BASE_URL}/aremodelsloaded")
    return res.json()

def is_server_running():
    print("[SERVER] Checking if server is running...")
    try:
        res = requests.get(f"{BASE_URL}/health")
        return res.status_code == 200
    except:
        return False

def loadIfNotLoaded(_configuration=None):
    # Starting backend for diffuse generation
    if is_server_running() == True:
        print("[INFO] Backend server is running.")
        if are_models_loaded() == False:
            print("[INFO] Loading models...")
            _load_all_models(_configuration)
        else:
            print("[INFO] Models are already loaded.")
    else:
        print("[INFO] Starting backend server...")
        start_server()
        
        print("[INFO] Waiting for server to become ready...")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            try:
                from PySide2.QtWidgets import QApplication
            except ImportError:
                QApplication = None

        retries = 30
        while retries > 0:
            if is_server_running():
                print("[INFO] Server is up and running!")
                break
            
            # Non-blocking wait for 1 second
            for _ in range(20):
                if QApplication and QApplication.instance():
                    QApplication.processEvents()
                time.sleep(0.05)
                
            retries -= 1
            
        if retries == 0:
            print("[ERROR] Server failed to start or respond in time.")
            return False
            
        print("[INFO] Loading models...")
        _load_all_models(_configuration)
        
        return True


def start_server():
    print("Starting server...")
    print("ENVIRONMENT_PATH: ", ENVIRONMENT_PATH)
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

