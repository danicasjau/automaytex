###############################################
## IMPORTING GENERAL LIBRARIES
################################################

import importlib
import maya.api.OpenMaya as om # type:ignore
import maya.cmds as cmds # type:ignore

try:
    ################################################
    ## IMPORTING PIPELINES
    ################################################

    import mPipline.geoExtraction.geometryRenderer
    import mPipline.geoExtractionSix.geoPlanarRenderer

    import mPipline.exrCollage.exrCollageGenerator
    import mPipline.exrCollage.exrCollageBroker

    import mPipline.mtlMaya.materialCreation
    import mPipline.uvUtils.reUvPorjection
    import mPipline.mtlMaya.mtlMaterialMapsCreation

    import mlGuiAdvanced

    import mlGui as mayagui
    import backServer
    import texPipeline

    print("All libraries imported successfully.")
except ImportError as e :
    print(f"[ERROR] Unable to start automaytex gui. \n Error occurred:\n{str(e)}")
    cmds.confirmDialog(
        title='Module AUTOMAAYTEX not found!', 
        message=f'Unable to start automaytex gui. \n\n Error occurred:\n{str(e)}', 
        button=['OK'], 
        defaultButton='OK', 
        icon='critical'
    )
    # Also log it to the Script Editor / Status Bar
    om.MGlobal.displayError(f"API Error: {str(e)}")

################################################
## RELOADING PIPELINES FOR DEVELOPING
################################################

def maya_remiport_libs():
    importlib.reload(mPipline.geoExtraction.geometryRenderer)
    importlib.reload(mPipline.geoExtractionSix.geoPlanarRenderer)

    importlib.reload(mPipline.exrCollage.exrCollageGenerator)
    importlib.reload(mPipline.exrCollage.exrCollageBroker)
    
    importlib.reload(mPipline.mtlMaya.materialCreation)
    importlib.reload(mPipline.uvUtils.reUvPorjection)
    importlib.reload(mPipline.mtlMaya.mtlMaterialMapsCreation)

    importlib.reload(mayagui)
    importlib.reload(backServer)

    importlib.reload(mPipline.geoExtractionSix.geoPlanarExtraction)
    importlib.reload(mPipline.geoExtractionSix.geoPlanarReProjectUV)
    importlib.reload(mPipline.geoExtractionSix.geoPlanarRenderer)
    importlib.reload(mlGuiAdvanced)
    importlib.reload(texPipeline)

maya_remiport_libs()

################################################
## AUTOMAYTEX PIPELINE EXECUTION
################################################

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
    # Use the progress update method from the global window
    progress_cb = _automaytex_window.update_progress if _automaytex_window else None
    pipeline = texPipeline.autoTexturePipeline(_configuration, progress_callback=progress_cb)
    pipeline.run()

start_gui()