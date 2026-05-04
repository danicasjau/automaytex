# maya automaytex plugin to load command
# command that generates the gui correspondent

import sys
import maya.api.OpenMaya as oM

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
        # importing code, and project
        from autotex import geoExtractionPipeline

        pipeline = geoExtractionPipeline()
        pipeline.run()

    
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
cmds.loadPlugin('sampleCommand.py')
cmds.myCommandName()

"""