# maya automaytex plugin to load command
# command that generates the gui correspondent


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
## SETTING UP CONFIGURATION from CONFIGURATION FILE
################################################

_BASE_PATH = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex"
_CONFIG_PATH = os.path.join(_BASE_PATH, "data", "configuration.json")

_config_data = {}
try:
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Config not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r") as _f:
        _config_data = json.load(_f)
except Exception as e:
    print(f"Warning: Could not load default configuration. Prompting user... ({e})")
    _selected_path = get_config_path_gui(_CONFIG_PATH)
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

################################################
## FUNCTION TO OPEN CONFIGURATION FILE - IF NOT FOUND
################################################

def get_config_path_gui(initial_path=""):
    maya_win = next((w for w in QtWidgets.QApplication.topLevelWidgets() if w.objectName() == "MayaWindow"), None)
    
    dialog = QtWidgets.QDialog(maya_win)
    dialog.setWindowTitle("Automaytex | Configuration Selector")
    dialog.setMinimumWidth(500)
    
    layout = QtWidgets.QVBoxLayout(dialog)
    path_input = QtWidgets.QLineEdit(initial_path)
    
    def browse():
        start_dir = os.path.dirname(initial_path) if os.path.exists(initial_path) else None
        res = cmds.fileDialog2(ff="JSON Files (*.json)", ds=2, fm=1, cap="Select Config", dir=start_dir)
        if res: path_input.setText(res[0])

    layout.addWidget(QtWidgets.QLabel("Configuration JSON Path"))
    
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
# Sample usage.
##########################################################

"""
# example on how to load plugin maya 

import maya.cmds as cmds
cmds.loadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')

cmds.reloadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')

cmds.unloadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')
cmds.unloadPlugin('automaytex.py')


import maya.cmds as cmds
cmds.loadPlugin('automaytex.py')
cmds.automaytex()



cmds.unloadPlugin('automaytex.py')
cmds.loadPlugin(r'D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\automaytex.py')
cmds.automaytex()
"""