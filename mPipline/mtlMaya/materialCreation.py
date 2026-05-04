import maya.cmds as cmds
import maya.mel as mel
import os

class autoMaMaterial:
    def __init__(self, config=None):
        if config is None:
            from config import configuration
            config = configuration()
        self.config = config
        self.mName = "empty_autoMaterial"
        self.mType = "standardSurface"

        self.mObject = None
        self.material = None

    def create(self):
        self.material = self._create_material(self.mName)
        self.shading_group = self._create_shading_group()

    def _create_material(self, name):
        mat = cmds.shadingNode("standardSurface", asShader=True)
        mat = cmds.rename(mat, name)
        return mat

    def _create_shading_group(self):
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=self.material + "SG")
        cmds.connectAttr(self.material + ".outColor", sg + ".surfaceShader", force=True)
        return sg

    def connectImage(self, image_path, slot="diffuse", udim=False):
        if not os.path.exists(image_path):
            cmds.warning(f"Texture not found: {image_path}")
            return

        file_node = cmds.shadingNode("file", asTexture=True, isColorManaged=True)
        place2d = cmds.shadingNode("place2dTexture", asUtility=True)

        # Connect place2d to file node
        for attr in [
            "coverage", "translateFrame", "rotateFrame", "mirrorU", "mirrorV",
            "stagger", "wrapU", "wrapV", "repeatUV", "offset", "rotateUV",
            "noiseUV", "vertexUvOne", "vertexUvTwo", "vertexUvThree",
            "vertexCameraOne"
        ]:
            cmds.connectAttr(place2d + "." + attr, file_node + "." + attr, force=True)

        cmds.connectAttr(place2d + ".outUV", file_node + ".uvCoord", force=True)
        cmds.connectAttr(place2d + ".outUvFilterSize", file_node + ".uvFilterSize", force=True)
        cmds.setAttr(file_node + ".fileTextureName", image_path, type="string")
        
        if udim:
            # Set UDIM tiling mode (Mari convention = 3)
            cmds.setAttr(file_node + ".uvTilingMode", 3)
            mel.eval('generateAllUvTilePreviews;')

        if slot == "normal":
            # Using Arnold's aiNormalMap instead of standard bump2d for better Arnold integration
            normal_map_node = cmds.shadingNode("aiNormalMap", asUtility=True)
            cmds.connectAttr(file_node + ".outColor", normal_map_node + ".input", force=True)
            cmds.connectAttr(normal_map_node + ".outValue", self.material + ".normalCamera", force=True)
            
            try: 
                cmds.setAttr(file_node + ".colorSpace", "Raw", type="string")
            except: 
                pass
            return

        if slot == "height":
            disp_node = cmds.shadingNode("displacementShader", asShader=True)
            cmds.setAttr(disp_node + ".scale", 0.01)
            cmds.connectAttr(file_node + ".outAlpha", disp_node + ".displacement", force=True)
            cmds.connectAttr(disp_node + ".displacement", self.shading_group + ".displacementShader", force=True)
            try: cmds.setAttr(file_node + ".colorSpace", "Raw", type="string")
            except: pass
            return

        slot_map = {
            "diffuse": (".baseColor", True),
            "albedo": (".baseColor", True),
            "roughness": (".specularRoughness", False),
            "metalness": (".metalness", False),
            "opacity": (".opacity", False),
        }

        if slot not in slot_map:
            cmds.warning(f"Unsupported slot: {slot}")
            return

        attr, is_color = slot_map[slot]

        if not is_color:
            try: cmds.setAttr(file_node + ".colorSpace", "Raw", type="string")
            except: pass

        if is_color:
            cmds.connectAttr(file_node + ".outColor", self.material + attr, force=True)
        else:
            # For grayscale, often outColorR or outAlpha is used. In Maya standardSurface it's usually outColorR or outAlpha
            cmds.connectAttr(file_node + ".outAlpha", self.material + attr, force=True)

    def connectImages(self, image_dict, udim=False):
        for slot, image_path in image_dict.items():
            if image_path:
                self.connectImage(image_path, slot=slot, udim=udim)

    def assign_to_object(self):
        if not self.mObject:
            cmds.warning("No objects to assign material to.")
            return
        cmds.sets(self.mObject, edit=True, forceElement=self.shading_group)


if __name__ == "__main__":
    mat_creator = autoMaMaterial()
    mat_creator.mName = "myAutoMaterial"
    mat_creator.mObject = cmds.ls(selection=True)
    mat_creator.create()
    mat_creator.connectImage(r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\images\09.jpg", slot="diffuse")
    mat_creator.assign_to_object()