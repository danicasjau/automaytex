"""
AUTOMAYTEX - AUTOMATIC TEXTURING TOOL FOR MAYA
Author: Daniel Casadevall

AutMayTex es un plugin de Maya que serveix per a texturitzar una mesh simple, via un prompt.

AutoMayTex renderitza una mesh (Depth i normal map) des de múltiples angles, 
introdueix les imatges renderitzades en un pipeline de Stable Diffusion XL ControlNet, 
i aplica les textures PBR resultants a la mesh a Maya.

Funcionalitats clau:
- Renderitzat de Depth i Normal maps des de múltiples angles.
- Integració amb Stable Diffusion XL ControlNet per a generar textures PBR.
- Aplicació automàtica de les textures generades a la mesh a Maya.

Distribució dels arxius:
- automaytex.py: Plugin principal de Maya que gestiona la integració i el workflow.
- command.py: Conté la lògica per a iniciar la GUI i gestionar les interaccions de l'usuari.
- config.py: Gestiona la configuració del pipeline, incloent rutes i paràmetres.

- mlGui.py: la implementació de la GUI per a configurar i executar el pipeline dins de Maya.
- mlGuiAdvanced.py: una versió avançada de la GUI amb funcionalitats addicionals per a usuaris avançats.

"""

################################################
## IMPORTING GENERAL LIBRARIES
################################################

import sys
import os
import json

import maya.api.OpenMaya as oM # type: ignore
import maya.cmds as cmds # type: ignore

try:
    from PySide6 import QtWidgets, QtCore, QtGui # type: ignore
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui # type: ignore


################################################
## FUNCTION TO OPEN CONFIGURATION FILE - IF NOT FOUND
################################################

def get_config_path_gui(initial_path=""):
    maya_win = next((w for w in QtWidgets.QApplication.topLevelWidgets() if w.objectName() == "MayaWindow"), None)
    
    dialog = QtWidgets.QDialog(maya_win)
    dialog.setWindowTitle("Automaytex | Configuration Selector")
    dialog.setMinimumWidth(500)
    
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setSpacing(15)  # Add breathing room between sections
    
    # --- NEW: Centered Icon & Welcome Message ---
    welcome_layout = QtWidgets.QVBoxLayout()
    welcome_layout.setAlignment(QtCore.Qt.AlignCenter)
    
    # Centered Top Icon (Uses standard Maya info icon, or you can pass a custom path)
    icon_label = QtWidgets.QLabel()
    icon_style = dialog.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation)
    icon_label.setPixmap(icon_style.pixmap(48, 48))  # 48x48 pixels
    icon_label.setAlignment(QtCore.Qt.AlignCenter)
    welcome_layout.addWidget(icon_label)
    
    # Welcome Title Message
    welcome_title = QtWidgets.QLabel("Welcome to Automaytex")
    font = welcome_title.font()
    font.setPointSize(14)
    font.setBold(True)
    welcome_title.setFont(font)
    welcome_title.setAlignment(QtCore.Qt.AlignCenter)
    welcome_layout.addWidget(welcome_title)
    
    # Welcome Subtitle
    welcome_sub = QtWidgets.QLabel("Please select or verify your configuration folder path below.")
    welcome_sub.setStyleSheet("color: #aaaaaa;")  # Subtle grey text for Maya's dark theme
    welcome_sub.setAlignment(QtCore.Qt.AlignCenter)
    welcome_layout.addWidget(welcome_sub)
    
    layout.addLayout(welcome_layout)
    
    # Thin divider line
    divider = QtWidgets.QFrame()
    divider.setFrameShape(QtWidgets.QFrame.HLine)
    divider.setFrameShadow(QtWidgets.QFrame.Sunken)
    layout.addWidget(divider)
    # --------------------------------------------
    
    path_input = QtWidgets.QLineEdit(initial_path)
    
    def browse():
        # Fallback handling if initial_path is empty
        if initial_path and os.path.isdir(initial_path):
            start_dir = initial_path
        elif initial_path:
            start_dir = os.path.dirname(initial_path)
        else:
            start_dir = os.path.expanduser("~") # Default to user home dir if empty

        res = cmds.fileDialog2(
            fm=3,  # folder mode
            ds=2,
            cap="Select Folder",
            dir=start_dir
        )

        if res:
            path_input.setText(res[0])

    layout.addWidget(QtWidgets.QLabel("Configuration Folder Path:"))
    
    row = QtWidgets.QHBoxLayout()
    row.addWidget(path_input)
    browse_btn = QtWidgets.QPushButton("Browse...")
    browse_btn.clicked.connect(browse)
    row.addWidget(browse_btn)
    layout.addLayout(row)

    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    btns.accepted.connect(dialog.accept)
    btns.rejected.connect(dialog.reject)
    layout.addWidget(btns)

    return path_input.text() if dialog.exec_() == QtWidgets.QDialog.Accepted else None


def edit_and_confirm_configuration_gui(config_json_path, new_base_dir, output_configuration_path="configuration.json"):
    """
    1. Loads existing config.
    2. Calculates new sub-paths automatically based on new_base_dir.
    3. Displays them in a GUI for the user to review, manually modify, or browse.
    4. Saves the results to output_configuration_path.
    """
    # --- STEP 1: READ EXISTING JSON ---
    if not os.path.exists(config_json_path):
        cmds.warning(f"Configuration template file not found at: {config_json_path}")
        return None

    try:
        with open(config_json_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        cmds.error(f"Failed to read JSON setup template: {e}")
        return None

    # --- STEP 2: CALCULATE AUTOMATIC PATHS ---
    # Update keys automatically based on your directory infrastructure logic
    config_data["BASE_DIR"] = new_base_dir
    config_data["ENV_PATH"] = os.path.join(new_base_dir, "mEnv")
    config_data["SCRIPTS_PATH"] = os.path.join(new_base_dir, "mPipline")
    config_data["MODELS_PATH"] = os.path.join(new_base_dir, "models")

    # --- STEP 3: INITIALIZE THE GUI DISPLAYER / MODIFIER ---
    # Parent to Maya window safely to avoid window backgrounding
    maya_win = next((w for w in QtWidgets.QApplication.topLevelWidgets() if w.objectName() == "MayaWindow"), None)
    
    dialog = QtWidgets.QDialog(maya_win)
    dialog.setWindowTitle("Automaytex | Confirm Pipeline Paths")
    dialog.setMinimumWidth(550)
    
    layout = QtWidgets.QVBoxLayout(dialog)
    
    # Informative header
    info_lbl = QtWidgets.QLabel("<b>Calculated Paths:</b> Review, adjust if needed, and confirm.")
    layout.addWidget(info_lbl)
    layout.addSpacing(5)
    
    grid = QtWidgets.QGridLayout()
    layout.addLayout(grid)
    
    inputs = {}
    keys_to_edit = ["BASE_DIR", "ENV_PATH", "SCRIPTS_PATH", "MODELS_PATH"]
    
    # Populate UI rows with the updated path calculations
    for i, key in enumerate(keys_to_edit):
        label = QtWidgets.QLabel(f"{key}:")
        line_edit = QtWidgets.QLineEdit(str(config_data.get(key, "")))
        browse_btn = QtWidgets.QPushButton("Browse...")
        
        grid.addWidget(label, i, 0)
        grid.addWidget(line_edit, i, 1)
        grid.addWidget(browse_btn, i, 2)
        
        inputs[key] = line_edit
        
        # Scope closure to handle row-specific browse button behavior
        def make_browse_callback(edit_field=line_edit):
            def browse_folder():
                current_text = edit_field.text()
                start_dir = current_text if os.path.isdir(current_text) else os.path.dirname(config_json_path)
                
                res = cmds.fileDialog2(
                    fm=3,  # Target directories
                    ds=2,
                    cap="Select Directory Asset Path",
                    dir=start_dir
                )
                if res:
                    edit_field.setText(res[0])
            return browse_folder
            
        browse_btn.clicked.connect(make_browse_callback())

    layout.addSpacing(15)
    
    # Dialog Window Buttons
    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    btns.accepted.connect(dialog.accept)
    btns.rejected.connect(dialog.reject)
    layout.addWidget(btns)

    # --- STEP 4: EXECUTE DIALOG AND SAVE ON CONFIRM ---
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        # Collect final adjustments made by user in the text boxes
        for key in keys_to_edit:
            config_data[key] = inputs[key].text()
                
        # Resolve target path relative to the input file base directory
        config_dir = os.path.dirname(config_json_path)
        output_path = output_configuration_path
        
        try:
            with open(output_path, "w") as f:
                json.dump(config_data, f, indent=4)
            
            # Display green confirmation popup directly over Maya viewport view 
            success_msg = f"<span style='color:#a0ffa0;'>Configuration saved to {output_configuration_path} successfully!</span>"
            cmds.inViewMessage(amg=success_msg, pos="topCenter", fade=True)
            print(f"// Automaytex Saved Config: {output_path}")
            
        except Exception as e:
            cmds.error(f"Failed to save confirmed pipeline configuration: {e}")
            return None
            
        return config_data
        
    print("// Automaytex: Setup configuration canceled by user.")
    return None


################################################
## SETTING UP CONFIGURATION from CONFIGURATION FILE
################################################

_BASE_PATH = get_config_path_gui(os.path.dirname(os.getcwd())) # Set this to your project base path
_CONFIG_PATH = os.path.join(_BASE_PATH, "data", "configuration.json")
_RAW_CONFIG_PATH = os.path.join(_BASE_PATH, "data", "raw_configuration.json")
edit_and_confirm_configuration_gui(_CONFIG_PATH, _BASE_PATH, _CONFIG_PATH)

_config_data = {}

try:
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Config not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r") as _f:
        _config_data = json.load(_f)
except Exception as e:
    print(f"Warning: Could not load default configuration. Prompting user... ({e})")
    _selected_path = _CONFIG_PATH
    if _selected_path:
        _CONFIG_PATH = _selected_path
        try:
            with open(_CONFIG_PATH, "r") as _f:
                _config_data = json.load(_f)
        except Exception as e2:
            print(f"Error loading selected configuration: {e2}")
    else:
        print("Configuration loading cancelled by user.")

# Set environment variables for the pipeline
os.environ["BASE_DIR"] = str(_config_data.get("BASE_DIR", ""))
os.environ["ENV_PATH"] = str(_config_data.get("ENV_PATH", ""))
os.environ["SCRIPTS_PATH"] = str(_config_data.get("SCRIPTS_PATH", ""))
os.environ["MODELS_PATH"] = str(_config_data.get("MODELS_PATH", ""))

sys.path.append(_BASE_PATH)
sys.path.append(os.path.join(_config_data.get("ENV_PATH", ""), "lib", "site-packages"))

import config as conf

_general_configuration = conf.configuration()

##########################################################
# Plug-in 
##########################################################

class AutomaytexCommand(oM.MPxCommand):
    kPluginCmdName = 'automaytex'
    
    def __init__(self):
        oM.MPxCommand.__init__(self)
    
    @staticmethod 
    def cmdCreator():
        return AutomaytexCommand() 
    
    def doIt(self, args):
        # importing command to start gui
        import command as cm
        cm.start_gui()
    
##########################################################
# Plug-in initialization.
##########################################################

def maya_useNewAPI():
	pass

def initializePlugin(mobject):
    mautotexplugin = oM.MFnPlugin(mobject)
    try:
        mautotexplugin.registerCommand(AutomaytexCommand.kPluginCmdName, AutomaytexCommand.cmdCreator)
    except:
        sys.stderr.write('Failed to register command: ' + AutomaytexCommand.kPluginCmdName)

def uninitializePlugin(mobject):
    mautotexplugin = oM.MFnPlugin(mobject)
    try:
        mautotexplugin.deregisterCommand(AutomaytexCommand.kPluginCmdName)
    except:
        sys.stderr.write('Failed to unregister command: ' + AutomaytexCommand.kPluginCmdName)

##########################################################
# Sample maya usage
##########################################################

"""
import maya.cmds as cmds
cmds.unloadPlugin('automaytex.py')
cmds.loadPlugin('automaytex.py')
cmds.automaytex()
"""