
import sys
import os

print("""
###############################################################
########  LOADING AUTO TEXTURING MAYA PIPELINE SERVER  ########
###############################################################
""")

# Add project root to sys.path
sys.path.append(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\server")

from fastapi import FastAPI
from cModels import diffModels

app = FastAPI()
models = diffModels()

# Simple health check
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/generate")
def generate(prompt: str = "default"):
    return {
        "message": "generation complete",
        "prompt": prompt,
        "result": f"generated_texture_for_{prompt}"
    }

@app.get("/loadallmodels")
def load_all_models():
    print("Loading all models...")

    models.load_all()
    return True

@app.get("/unloadallmodels")
def unload_all_models():
    print("Unloading all models...")

    models.unload_all()
    return True