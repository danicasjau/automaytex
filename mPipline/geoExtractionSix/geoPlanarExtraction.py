import maya.cmds as cmds
import maya.mel as mel
import os
import math
import sys

sys.path.append(r"D:\DANI\PROJECTS_2026\autotexturingmaya\env\menv\lib\site-packages")

# ==============================================================
# GeometryPlanarExtractor
# Renders 6 orthographic views of selected geometry via Arnold.
# - Creates temporary lights for the render, deleted afterwards.
# - Renders a single multichannel EXR per view (Beauty, Depth, N).
# ==============================================================

class GeometryPlanarExtractor:

    # VIEW_ORDER and VIEW_ROTATIONS moved to config

    def __init__(self, export_path=None, resolution=None, config=None):
        if config is None:
            from config import configuration
            config = configuration()
        self.config       = config
        self.export_path  = export_path if export_path else self.config.temporal_path
        self.resolution   = resolution if resolution else self.config.resolution
        self.cam_name     = self.config.camera_name
        self.cam_shape    = None
        self._tmp_lights  = []   # track provisional lights for cleanup

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_transforms(self):
        sel = cmds.ls(sl=True, long=True)
        transforms = cmds.ls(sel, type="transform", long=True)
        if not transforms:
            shapes = cmds.ls(sel, dag=True, shapes=True, long=True)
            if shapes:
                transforms = list({
                    cmds.listRelatives(s, parent=True, fullPath=True)[0]
                    for s in shapes
                })
        if not transforms:
            raise RuntimeError("Nothing selected. Select at least one mesh transform.")
        return transforms

    def get_bounding_box(self, transforms):
        all_min = [ math.inf,  math.inf,  math.inf]
        all_max = [-math.inf, -math.inf, -math.inf]
        for tfm in transforms:
            bb = cmds.exactWorldBoundingBox(tfm)
            for i in range(3):
                all_min[i] = min(all_min[i], bb[i])
                all_max[i] = max(all_max[i], bb[i + 3])
        return all_min, all_max

    # ----------------------------------------------------------
    # Arnold init
    # ----------------------------------------------------------

    def _init_arnold(self):
        if not cmds.pluginInfo("mtoa", query=True, loaded=True):
            cmds.loadPlugin("mtoa", quiet=True)
        import mtoa.core as core
        core.createOptions()

    def _setup_aovs(self):
        """
        Register all required AOVs with Arnold:
          N - world-space normal
          Z - depth
        Beauty (RGBA) is rendered by default.
        """
        import mtoa.aovs as aovs
        ai = aovs.AOVInterface()
        existing = ai.getAOVNodes(names=True) or []

        standard_aovs = ["N", "Z"]
        for aov_name in standard_aovs:
            if aov_name not in existing:
                try:
                    ai.addAOV(aov_name)
                    print(f"  [aov] Added: {aov_name}")
                except Exception as e:
                    print(f"  [aov] Could not add '{aov_name}': {e}")

    # ----------------------------------------------------------
    # Provisional lighting rig and materials
    # ----------------------------------------------------------

    def _create_lights(self, bb_min, bb_max):
        method = 1
        if method == 1:
            """
            Create a temporary scalar-independent lighting rig: Skydome + Directional Light.
            Creates a uniform gray material and connects to Arnold's global shader override.
            """
            # Skydome 
            try:
                dome_tfm   = cmds.createNode("transform",      name="tmpSkyDome")
                dome_shape = cmds.createNode("aiSkyDomeLight", name="tmpSkyDomeShape", parent=dome_tfm)
                cmds.setAttr(dome_shape + ".intensity", 1.0)
                cmds.setAttr(dome_shape + ".camera", 0.0) # Hide from background
                self._tmp_lights.append(dome_tfm)
                print("  [light] Arnold skydome created.")
            except Exception:
                result = cmds.ambientLight(name="tmpSkyDome", intensity=1.0)
                self._tmp_lights.append(result[0] if isinstance(result, list) else result)
                print("  [light] ambientLight fallback created.")

            # Directional Key Light
            try:
                result = cmds.directionalLight(name="tmpDirLight", intensity=1.2)
                dir_tfm = result[0] if isinstance(result, list) else result
                cmds.setAttr(dir_tfm + ".rotateX", -45)
                cmds.setAttr(dir_tfm + ".rotateY", -45)
                self._tmp_lights.append(dir_tfm)
                print("  [light] directionalLight created.")
            except Exception as e:
                print("  [light] directionalLight failed:", e)
        else:
            # comment: 
        
            # 1. Calculate Scale & Distance based on Bounding Box
            # We use the diagonal or max dimension to ensure the lights aren't inside the mesh
            size = max([abs(bb_max[i] - bb_min[i]) for i in range(3)])
            dist = size * 2.5
            if dist == 0: dist = 10 # Fallback if BB is empty
            
            self._tmp_lights = []

            # --- 1. SKYDOME (Ambient Base) ---
            try:
                dome_tfm = cmds.createNode("transform", name="tmpSkyDome")
                dome_shape = cmds.createNode("aiSkyDomeLight", name="tmpSkyDomeShape", parent=dome_tfm)
                cmds.setAttr(f"{dome_shape}.intensity", 0.4) # Soft ambient
                cmds.setAttr(f"{dome_shape}.camera", 0.0)    # Hide from BG
                self._tmp_lights.append(dome_tfm)
            except:
                pass

            # --- 2. THE THREE POINTS ---
            # Setup Data: (Name, Intensity, Rotation_Y, Rotation_X, Exposure)
            light_specs = [
                ("KeyLight",  1.5,  35, -25, 1.0), # Main shape definer
                ("FillLight", 0.8, -45, -15, 0.5), # Softens shadows
                ("RimLight",  2.5, 180, -30, 2.0)  # Separates from background
            ]

            for name, intensity, rot_y, rot_x, exposure in light_specs:
                try:
                    # Create Arnold Area Light for soft, professional shadows
                    # Using aiAreaLight is better than standard directional for "render-ready" looks
                    light_node = cmds.shadingNode("aiAreaLight", asLight=True, name=f"tmp{name}Shape")
                    light_tfm = cmds.listRelatives(light_node, parent=True)[0]
                    
                    # Position the light
                    cmds.setAttr(f"{light_tfm}.rotateY", rot_y)
                    cmds.setAttr(f"{light_tfm}.rotateX", rot_x)
                    
                    # Move it back along its local Z axis
                    cmds.move(0, 0, dist, light_tfm, relative=True, objectSpace=True)
                    
                    # Attributes
                    cmds.setAttr(f"{light_node}.intensity", intensity)
                    cmds.setAttr(f"{light_node}.exposure", exposure)
                    cmds.setAttr(f"{light_node}.aiModifyBoundary", 1) # Normalize
                    
                    # Scale the area light so it produces soft shadows relative to model size
                    light_scale = size * 0.5
                    cmds.setAttr(f"{light_tfm}.scale", light_scale, light_scale, light_scale)
                    
                    self._tmp_lights.append(light_tfm)
                except Exception as e:
                    print(f"  [light] Failed to create {name}: {e}")

            print(f"  [light] Professional 3-point rig created at scale {dist:.2f}")



    def _override_shader(self):
        """Applies a neutral aiStandardSurface globally for preview renders."""
        try:
            shd = cmds.createNode("aiStandardSurface", name="_tmpPreviewShader")
            cmds.setAttr(shd + ".baseColor", 0.6, 0.6, 0.6, type="double3")
            cmds.setAttr(shd + ".specular", 0.2)
            cmds.setAttr(shd + ".specularRoughness", 0.5)

            self._orig_shader_override = None
            if cmds.objExists("defaultArnoldRenderOptions"):
                conns = cmds.listConnections("defaultArnoldRenderOptions.shader", plugs=True, connections=True)
                if conns:
                    self._orig_shader_override = conns[1]
                    cmds.disconnectAttr(conns[1], conns[0])
                
                cmds.connectAttr(shd + ".message", "defaultArnoldRenderOptions.shader", force=True)
                self._tmp_lights.append(shd)
                print("  [shader] Global preview shader applied.")
        except Exception as e:
            print("  [shader] Could not override shader:", e)

    def _restore_shader(self):
        """Restores the original global shader override, if any."""
        if hasattr(self, "_orig_shader_override") and cmds.objExists("defaultArnoldRenderOptions"):
            try:
                conns = cmds.listConnections("defaultArnoldRenderOptions.shader", plugs=True, connections=True)
                if conns:
                    cmds.disconnectAttr(conns[1], conns[0])
                if self._orig_shader_override and cmds.objExists(self._orig_shader_override.split('.')[0]):
                    cmds.connectAttr(self._orig_shader_override, "defaultArnoldRenderOptions.shader", force=True)
            except:
                pass

    def _delete_lights(self):
        for node in self._tmp_lights:
            if cmds.objExists(node):
                cmds.delete(node)
                print(f"  [cleanup] Deleted {node}")
        self._tmp_lights = []
        self._restore_shader()

    # ----------------------------------------------------------
    # Camera
    # ----------------------------------------------------------

    def _create_ortho_cam(self):
        if cmds.objExists(self.cam_name):
            cmds.delete(self.cam_name)
        cam_nodes      = cmds.camera(name=self.cam_name, orthographic=True)
        self.cam_name  = cam_nodes[0]
        self.cam_shape = cam_nodes[1]
        cmds.setAttr(self.cam_shape + ".nearClipPlane",  0.001)
        cmds.setAttr(self.cam_shape + ".farClipPlane",  100000)
        print(f"  [cam] Created orthographic camera: {self.cam_name}")

    def _frame_camera(self, view_name, bb_min, bb_max):
        rx, ry, rz = self.config.view_rotations[view_name]
        cx = (bb_min[0] + bb_max[0]) * 0.5
        cy = (bb_min[1] + bb_max[1]) * 0.5
        cz = (bb_min[2] + bb_max[2]) * 0.5

        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        dz = bb_max[2] - bb_min[2]

        if view_name in ("front", "back"):
            horiz, vert, depth_ext = dx, dy, dz
            axis = (0, 0,  1) if view_name == "front" else (0, 0, -1)
        elif view_name in ("left", "right"):
            horiz, vert, depth_ext = dz, dy, dx
            axis = (1, 0, 0) if view_name == "left"  else (-1,  0,  0)
        else:  # top / bottom
            horiz, vert, depth_ext = dx, dz, dy
            axis = (0,  1, 0) if view_name == "top"   else (0, -1,  0)

        padding = self.config.ortho_padding
        ortho_w = max(horiz, vert) * (1.0 + padding)
        dist    = max(depth_ext * 0.5 + 1.0, 10.0)

        cmds.setAttr(self.cam_name + ".translateX", cx + axis[0] * dist)
        cmds.setAttr(self.cam_name + ".translateY", cy + axis[1] * dist)
        cmds.setAttr(self.cam_name + ".translateZ", cz + axis[2] * dist)
        cmds.setAttr(self.cam_name + ".rotateX",    rx)
        cmds.setAttr(self.cam_name + ".rotateY",    ry)
        cmds.setAttr(self.cam_name + ".rotateZ",    rz)
        cmds.setAttr(self.cam_shape + ".orthographicWidth", ortho_w)

    # ----------------------------------------------------------
    # Render one view → merged EXR
    # ----------------------------------------------------------

    def _set_render_globals_exr(self):
        """Configure Arnold for EXR output (full float, merged AOVs)."""
        cmds.setAttr("defaultResolution.width",             self.resolution)
        cmds.setAttr("defaultResolution.height",            self.resolution)
        cmds.setAttr("defaultResolution.deviceAspectRatio", 1.0)
        cmds.setAttr("defaultArnoldRenderOptions.aovMode",  1)  # AOVs ON
        cmds.setAttr("defaultArnoldDriver.aiTranslator", "exr", type="string")
        cmds.setAttr("defaultArnoldDriver.mergeAOVs", 1)  # Merge into single EXR

    def _render_view(self, view_name, bb_min, bb_max):
        """
        Renders a single EXR per plane containing Beauty, Depth (Z), and Normal (N).
        """
        self._frame_camera(view_name, bb_min, bb_max)

        self._set_render_globals_exr()
        # No <RenderPass> token in prefix because we merge everything into one file
        exr_prefix = os.path.join(self.export_path, view_name)
        cmds.setAttr("defaultArnoldDriver.prefix", exr_prefix, type="string")
        print(f"  [render] {view_name} merged AOV EXR ...")
        cmds.arnoldRender(width=self.resolution, height=self.resolution,
                          camera=self.cam_shape, batch=True)

        exr_file = os.path.join(self.export_path, f"{view_name}.exr")
        return {"exr": exr_file}

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------

    def run(self):
        self.ensure_dir(self.export_path)
        self._init_arnold()

        # Register all AOVs (N, Z, diffuse_color, AO, P, crypto_*)
        self._setup_aovs()

        transforms = self.get_transforms()
        bb_min, bb_max = self.get_bounding_box(transforms)
        print(f"  [bbox] min={bb_min}  max={bb_max}")

        # Provisional lighting and preview material
        self._create_lights(bb_min, bb_max)
        self._override_shader()
        self._create_ortho_cam()

        # Render all 6 views
        all_passes = {}
        for view_name in self.config.face_order:
            print(f"\n[view] === {view_name.upper()} ===")
            all_passes[view_name] = self._render_view(view_name, bb_min, bb_max)

        # Cleanup camera and provisional lights
        if cmds.objExists(self.cam_name):
            cmds.delete(self.cam_name)
        self._delete_lights()

        print("\n=== GeometryPlanarExtractor DONE ===")
        print(f"  Output folder : {self.export_path}")



# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    extractor = GeometryPlanarExtractor(
        export_path=r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\automaytex\temp",
        resolution=512
    )
    extractor.run()