
import sys
import os

print("""
###############################################################
########  LOADING AUTO TEXTURING MAYA PIPELINE SERVER  ########
###############################################################
""")

# Add project root to sys.path
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\server")

from fastapi import FastAPI, Body
from cModels import diffModels
from diffgenSDXL import DiffGenSDXL

app = FastAPI()
models = diffModels()

# Simple health check
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/generatetexture")
def generate_texture(configuration: dict = Body(...)):

    pipeGen = DiffGenSDXL(models, configuration)
    saved_path = pipeGen.generate()

    return {
        "state": "success",
        "message": "generation complete",
        "output_path": saved_path,
    }

@app.post("/loadallmodels")
def load_all_models(configuration: dict = Body(...)):
    print("Loading all models...")

    models.load_all(configuration)
    return True

@app.get("/unloadallmodels")
def unload_all_models():
    print("Unloading all models...")

    models.unload_all()
    return True