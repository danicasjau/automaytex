
import sys
import os

print("""
###############################################################
########  LOADING AUTO TEXTURING MAYA PIPELINE SERVER  ########
###############################################################
""")

# Add project root to sys.path
sys.path.append(os.path.join(os.environ.get('BASE_DIR'), "server"))

from fastapi import FastAPI, Body
from cModels import diffModels
from diffgenSDXL import DiffGenSDXL

app = FastAPI()
models = diffModels()

# Simple health check
@app.get("/health")
def health():
    print("[SERVER] Health check")
    return {"status": "ok"}

@app.get("/aremodelsloaded")
def are_models_loaded():
    print("[SERVER] Checking if models are loaded...")
    return models.are_all_loaded()



@app.post("/generatetexture")
def generate_texture(configuration: dict = Body(...)):
    print("[SERVER] Generating texture...")
    pipeGen = DiffGenSDXL(models, configuration)
    saved_path = pipeGen.generate()

    return {
        "state": "success",
        "message": "generation complete",
        "output_path": saved_path,
    }

@app.post("/loadallmodels")
def load_all_models(configuration: dict = Body(...)):
    print("[SERVER] Loading all models...")

    models.load_all(configuration)
    return True

@app.get("/unloadallmodels")
def unload_all_models():
    print("[SERVER] Unloading all models...")

    models.unload_all()
    return True