# AutoMayTex

**AI-driven automatic texture generation for Autodesk Maya.** AutoMayTex renders a mesh from multiple angles, feeds the renders into a Stable Diffusion XL ControlNet pipeline, and automatically applies the resulting PBR texture maps back onto the mesh.

---

## What It Does

AutoMayTex takes a selected Maya mesh and runs a fully automated pipeline:

1. **Geometry Rendering** — Renders the mesh from 4 tetrahedral viewpoints into EXR tiles (normals + depth + RGBA).
2. **Collage Baking** — Tiles are packed into a 2×2 collage for efficient inference.
3. **AI Texture Generation** — The collage is sent to a local FastAPI server running a Stable Diffusion XL ControlNet model. The model generates diffuse, roughness, metalness, normal, and height maps guided by your prompt.
4. **UV Retargeting** — Generated tiles are re-projected back onto the mesh's original UV layout using planar UV baking.
5. **Material Assignment** — PBR maps are automatically wired into an Arnold/MaterialX/Standard Surface material and assigned to the object.

Everything is orchestrated from a non-blocking PySide6 GUI embedded inside Maya, with a real-time progress bar. The AI inference runs in a separate Python venv to avoid conflicts with Maya's internal Python interpreter.

---

## Pipeline Architecture

```
Maya Mesh
    │
    ▼
[Geometry Renderer]          ← Tetrahedral camera rig, EXR output
    │
    ▼
[EXR Collage Generator]      ← 2×2 tile collage
    │
    ▼
[FastAPI Backend Server]     ← SDXL + ControlNet (Normal/Depth guided)
    │   (port 8001, separate venv)
    ▼
[Diffusion Output]           ← diffuse / roughness / metalness / normal / height PNGs
    │
    ▼
[UV Retarget + Material]     ← Re-projects tiles, creates & assigns Maya material
```

The backend server (`server/server.py`) runs inside a dedicated Python virtual environment with PyTorch + Diffusers. Maya communicates with it over HTTP on `localhost:8001`. The server is started automatically by the plugin when needed.

---

## Supported Models

AutoMayTex currently supports three SDXL-family base models:

| ID | Model | Best For |
|---|---|---|
| `sdxl` | Juggernaut XL v9 | High quality, photo-realistic textures |
| `fast_sdxl` | SDXL Lightning 4-step | Fast previews — **recommended steps: 4, CFG: 2.0** |
| `flash_sdxl` | SDXL Lightning 1-step | Ultra-fast generation |

All models require a **ControlNet Union SDXL** model for normal-guided generation, and a **Depth Anything ViT-L14** model for depth estimation. Model paths are configured in `data/models.json`.

---

## Installation

### Option A — GUI Installer (Recommended)

1. Run `installer/installer_gui.bat` by double-clicking it.
2. The wizard will guide you through:
   - Detecting your Maya installation
   - Choosing an installation folder
   - Cloning the repository from GitHub
   - Optionally downloading AI models
   - Installing Python dependencies
   - Registering the plug-in with Maya
3. Open Maya → **Windows → Settings/Preferences → Plug-in Manager**
4. Find `automayatex.py`, enable **Loaded** and **Auto load**.

> Requires Git for Windows: https://git-scm.com/download/win

---

### Option B — Command-Line Installer

1. Run `installer.bat` from the repository root.
2. Follow the interactive prompts to select your Maya version, installation path, and optional model download.

---

### Option C — Manual Installation

#### 1. Clone the repository

```bash
git clone https://github.com/danicasjau/automaytex.git
cd automaytex
```

#### 2. Create a Python virtual environment for the backend

Use a standalone Python 3.10–3.12 (not Maya's Python) to create the backend venv:

```bash
python -m venv mEnv
mEnv\Scripts\pip install --upgrade pip
mEnv\Scripts\pip install -r requirements.txt
```

> For CUDA GPU acceleration (strongly recommended), make sure you have CUDA 12.4+ installed. The `requirements.txt` already points to the correct PyTorch CUDA wheels.

#### 3. Configure `configuration.json`

Create or edit `data/configuration.json` to set your paths:

```json
{
  "BASE_DIR": "D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex",
  "ENV_PATH": "D:/DANI/PROJECTS_2026/AutoTexturingMaya/mEnv",
  "SCRIPTS_PATH": "...",
  "MODELS_PATH": "D:/DANI/PROJECTS_2026/AutoTexturingMaya/automaytex/models"
}
```

These paths are loaded as environment variables by `automaytex.py` on plugin startup.

#### 4. Load the plugin in Maya

Open the **Maya Script Editor** and run:

```python
import maya.cmds as cmds

# Load the plugin from its full path
cmds.loadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')

# Launch the GUI
cmds.automaytex()
```

To **reload** the plugin during development:

```python
import maya.cmds as cmds
cmds.unloadPlugin('automaytex.py')
cmds.loadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')
cmds.automaytex()
```

#### 5. Alternative: Add to Maya's plugin path

Instead of specifying the full path every time, add the plugin folder to Maya's plugin search path. In Maya's **Script Editor** run once:

```python
import maya.cmds as cmds
import os
os.environ["MAYA_PLUG_IN_PATH"] = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"
```

Or set `MAYA_PLUG_IN_PATH` permanently as a Windows environment variable, then reload Maya and find `automaytex.py` in **Plug-in Manager**.

---

## Loading AI Models

### From the GUI (Recommended)

1. Launch AutoMayTex via `cmds.automaytex()`.
2. Click **Advanced Model Settings**.
3. Go to the **Models** tab.
4. For each model, verify or change the **Install path**.
5. Click **Install / Download** on any missing model — the download runs in the background with a live progress bar.
6. Once all models are installed, go to the **Server** tab and click **Start Server**, then **Load Models**.

### Manual Model Download

You can also download model files manually from HuggingFace and place them in the paths configured in `data/models.json`:

| Model | Source | Destination Key |
|---|---|---|
| Juggernaut XL v9 | [HuggingFace](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/juggernautXL_v9Rdphoto2Lightning.safetensors) | `installation_path` for `sdxl` |
| SDXL Lightning 4-step | [HuggingFace](https://huggingface.co/ByteDance/SDXL-Lightning) | `installation_path` for `fast_sdxl` |
| ControlNet Union SDXL | [HuggingFace](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promaxx.safetensors) | `installation_path` for `controlnet` |
| Depth Anything ViT-L14 | [HuggingFace](https://huggingface.co/LiheYoung/depth-anything-vitl14/tree/main) *(snapshot)* | `installation_path` for `depth` |

After downloading, update `data/models.json` to point `installation_path` at the folder containing each file. The Depth Anything model is a full repository — use `huggingface-cli download` or `snapshot_download()`:

```bash
pip install huggingface_hub
huggingface-cli download LiheYoung/depth-anything-vitl14 --local-dir ./models/depth_anything_vitl14
```

### Verifying Model Paths

Open `data/models.json` to confirm paths are correct:

```json
{
  "models": [
    {
      "name": "sdxl",
      "installation_path": "E:/models/checkpoints/",
      "installation_name": "juggernautXL_v9Rdphoto2Lightning.safetensors"
    },
    {
      "name": "controlnet",
      "installation_path": "E:/models/controlnet/",
      "installation_name": "diffusion_pytorch_model_promaxx.safetensors"
    },
    {
      "name": "depth",
      "installation_path": "D:/models/",
      "installation_name": "depth_anything_vitl14"
    }
  ]
}
```

---

## Usage

1. Select a mesh in Maya.
2. Open the plugin: `cmds.automaytex()` (or via the Plug-in shelf).
3. Enter a **Positive Prompt** (e.g. `"rusted medieval iron armor, seamless, 8K"`).
4. Set the **Material Name** and optionally a custom **Output Path**.
5. Select which **Maps** to generate (Diffuse, Roughness, Metalness, Normal, Height).
6. Choose your **Model Type**. For `fast_sdxl`, use **Steps: 4** and **CFG: 2.0**.
7. Click **Generate Textures**.
8. The progress bar tracks each pipeline stage. A popup confirms when done.

---

## Requirements

- **Autodesk Maya** 2024 / 2025 / 2026 (PySide6)
- **NVIDIA GPU** with CUDA 12.4+ (strongly recommended; CPU supported but slow)
- **Python** 3.10 – 3.12 (standalone, for the backend venv)
- **Git for Windows** (for installer)
- ~**20 GB disk** for full model set

---

## Credits

AutoMayTex — v1.2.0  
Daniel Casadevall Jauhiainen — La Salle, 2026
