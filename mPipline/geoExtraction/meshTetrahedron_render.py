# ==============================================================
# GeometryPlanarExtractor
# Renders 4 orthographic views from the face normals of a
# regular tetrahedron enclosing the selected geometry.
# - Creates temporary lights for the render, deleted afterwards.
# - Renders a single multichannel EXR per view (Beauty, Depth, N).
# ==============================================================

import maya.cmds as cmds
import maya.mel as mel

import mtoa.core as core
import mtoa.aovs as aovs

import os
import math
import sys

# ----------------------------------------------------------
# Regular tetrahedron face normals (outward, unit vectors)
# and matching camera rotations (rx, ry, rz in degrees).
#
# A regular tetrahedron has 4 equilateral triangle faces.
# Each camera is placed along the outward normal of one face,
# pointing toward the centroid, so together they cover the
# object from all sides with no redundant orthogonal views.
#
# Normals derived from a canonical tetrahedron inscribed in
# the unit sphere with one face flat on the bottom:
#
#   v0 = ( 0,       1,        0     )   ← top vertex
#   v1 = ( 2√2/3,  -1/3,      0     )
#   v2 = (-√2/3,   -1/3,  √(2/3)  )
#   v3 = (-√2/3,   -1/3, -√(2/3)  )
#
# Face normals (inward → flip for camera direction):
#   face_0 (opposite v0, bottom) :  ( 0,     -1,      0    )
#   face_1 (opposite v1)         :  (-2√2/3,  1/3,    0    )  normalised
#   face_2 (opposite v2)         :  ( √2/3,   1/3,  -√(2/3)) normalised
#   face_3 (opposite v3)         :  ( √2/3,   1/3,   √(2/3)) normalised
#
# Maya camera Euler conventions (intrinsic XYZ, degrees):
#   rx = -arcsin(ny)          (pitch: negative = looking up)
#   ry =  arctan2(nx, nz)     (yaw around world Y)
#   rz = 0                    (no roll needed)
# ----------------------------------------------------------

def _tetrahedron_views():
    """
    Returns a dict of {view_name: (rx, ry, rz, axis_unit_vector)}
    for the 4 face normals of a regular tetrahedron.
    Camera placed along the outward normal, aimed at centroid.
    """
    s2 = math.sqrt(2)
    s23 = math.sqrt(2.0 / 3.0)

    # Outward face normals (unit vectors pointing away from centroid)
    normals = {
        "face_0": ( 0.0,            -1.0,       0.0   ),   # bottom face
        "face_1": (-2*s2/3,          1.0/3,     0.0   ),   # front-right
        "face_2": ( s2/3,            1.0/3,    -s23   ),   # back-right
        "face_3": ( s2/3,            1.0/3,     s23   ),   # back-left
    }

    views = {}
    for name, (nx, ny, nz) in normals.items():
        # Normalise (already unit, but guard against float drift)
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        nx, ny, nz = nx/length, ny/length, nz/length

        # Camera Euler angles so it looks from (nx,ny,nz) toward origin
        rx = math.degrees(-math.asin(max(-1.0, min(1.0, ny))))
        ry = math.degrees(math.atan2(nx, nz))
        rz = 0.0

        views[name] = (rx, ry, rz, (nx, ny, nz))

    return views


class GeometryPlanarExtractor:
    def __init__(self, export_path=None, resolution=None):

        self.export_path = export_path
        self.resolution  = resolution
        self.cam_name    = "faceCamera"
        self.cam_shape   = None

        # Pre-compute the 4 tetrahedral views
        self._tet_views  = _tetrahedron_views()

        self._tmp_lights = []

    # ----------------------------------------------------------
    # Utilities
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
        core.createOptions()

    def _setup_aovs(self):
        ai = aovs.AOVInterface()
        existing = ai.getAOVNodes(names=True) or []
        for aov_name in ("N", "Z"):
            if aov_name not in existing:
                try:
                    ai.addAOV(aov_name)
                    print(f"  [aov] Added: {aov_name}")
                except Exception as e:
                    print(f"  [aov] Could not add '{aov_name}': {e}")

    # ----------------------------------------------------------
    # Provisional lighting rig
    # ----------------------------------------------------------

    def _create_lights(self, bb_min, bb_max):
        """
        Skydome + directional light.  Same as original – lighting is
        view-independent so one rig covers all 4 tetrahedral cameras.
        """
        try:
            dome_tfm   = cmds.createNode("transform",      name="tmpSkyDome")
            dome_shape = cmds.createNode("aiSkyDomeLight", name="tmpSkyDomeShape",
                                         parent=dome_tfm)
            cmds.setAttr(dome_shape + ".intensity", 1.0)
            cmds.setAttr(dome_shape + ".camera",    0.0)
            self._tmp_lights.append(dome_tfm)
            print("  [light] Arnold skydome created.")
        except Exception:
            result = cmds.ambientLight(name="tmpSkyDome", intensity=1.0)
            self._tmp_lights.append(result[0] if isinstance(result, list) else result)
            print("  [light] ambientLight fallback created.")

        try:
            result  = cmds.directionalLight(name="tmpDirLight", intensity=1.2)
            dir_tfm = result[0] if isinstance(result, list) else result
            cmds.setAttr(dir_tfm + ".rotateX", -45)
            cmds.setAttr(dir_tfm + ".rotateY", -45)
            self._tmp_lights.append(dir_tfm)
            print("  [light] directionalLight created.")
        except Exception as e:
            print("  [light] directionalLight failed:", e)

    def _override_shader(self):
        try:
            shd = cmds.createNode("aiStandardSurface", name="_tmpPreviewShader")
            cmds.setAttr(shd + ".baseColor",         0.6, 0.6, 0.6, type="double3")
            cmds.setAttr(shd + ".specular",          0.2)
            cmds.setAttr(shd + ".specularRoughness", 0.5)

            self._orig_shader_override = None
            if cmds.objExists("defaultArnoldRenderOptions"):
                conns = cmds.listConnections(
                    "defaultArnoldRenderOptions.shader",
                    plugs=True, connections=True)
                if conns:
                    self._orig_shader_override = conns[1]
                    cmds.disconnectAttr(conns[1], conns[0])
                cmds.connectAttr(shd + ".message",
                                 "defaultArnoldRenderOptions.shader", force=True)
                self._tmp_lights.append(shd)
                print("  [shader] Global preview shader applied.")
        except Exception as e:
            print("  [shader] Could not override shader:", e)

    def _restore_shader(self):
        if hasattr(self, "_orig_shader_override") and \
                cmds.objExists("defaultArnoldRenderOptions"):
            try:
                conns = cmds.listConnections(
                    "defaultArnoldRenderOptions.shader",
                    plugs=True, connections=True)
                if conns:
                    cmds.disconnectAttr(conns[1], conns[0])
                node = (self._orig_shader_override or "").split('.')[0]
                if self._orig_shader_override and cmds.objExists(node):
                    cmds.connectAttr(self._orig_shader_override,
                                     "defaultArnoldRenderOptions.shader",
                                     force=True)
            except Exception:
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
        cam_nodes     = cmds.camera(name=self.cam_name, orthographic=True)
        self.cam_name = cam_nodes[0]
        self.cam_shape = cam_nodes[1]
        cmds.setAttr(self.cam_shape + ".nearClipPlane",  0.001)
        cmds.setAttr(self.cam_shape + ".farClipPlane",  100000)
        print(f"  [cam] Created orthographic camera: {self.cam_name}")

    def _frame_camera_tetrahedral(self, view_name, bb_min, bb_max):
        """
        Place the orthographic camera along the outward face normal of a
        regular tetrahedron, aimed at the bounding-box centroid.

        The orthographic width is set to the diagonal of the bounding
        sphere so the object always fits regardless of viewing angle,
        with an optional padding factor.
        """
        rx, ry, rz, (nx, ny, nz) = self._tet_views[view_name]

        # Bounding-box centroid
        cx = (bb_min[0] + bb_max[0]) * 0.5
        cy = (bb_min[1] + bb_max[1]) * 0.5
        cz = (bb_min[2] + bb_max[2]) * 0.5

        # Bounding-sphere radius (half-diagonal of the BB)
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        dz = bb_max[2] - bb_min[2]
        bb_radius = 0.5 * math.sqrt(dx*dx + dy*dy + dz*dz)
        if bb_radius == 0.0:
            bb_radius = 1.0   # degenerate / empty selection guard

        # Camera distance: enough to be outside the bounding sphere
        dist = bb_radius + max(bb_radius * 0.5, 10.0)

        # Camera position along outward normal from centroid
        cam_x = cx + nx * dist
        cam_y = cy + ny * dist
        cam_z = cz + nz * dist

        # Orthographic width = bounding-sphere diameter × padding
        ortho_w  = bb_radius * 2.0 * (1.0 + 0.1)

        cmds.setAttr(self.cam_name + ".translateX", cam_x)
        cmds.setAttr(self.cam_name + ".translateY", cam_y)
        cmds.setAttr(self.cam_name + ".translateZ", cam_z)
        cmds.setAttr(self.cam_name + ".rotateX",    rx)
        cmds.setAttr(self.cam_name + ".rotateY",    ry)
        cmds.setAttr(self.cam_name + ".rotateZ",    rz)
        cmds.setAttr(self.cam_shape + ".orthographicWidth", ortho_w)

        print(f"    cam pos  : ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")
        print(f"    cam rot  : rx={rx:.1f}  ry={ry:.1f}  rz={rz:.1f}")
        print(f"    ortho_w  : {ortho_w:.3f}")

    # ----------------------------------------------------------
    # Render
    # ----------------------------------------------------------

    def _set_render_globals_exr(self):
        cmds.setAttr("defaultResolution.width",             self.resolution)
        cmds.setAttr("defaultResolution.height",            self.resolution)
        cmds.setAttr("defaultResolution.deviceAspectRatio", 1.0)
        cmds.setAttr("defaultArnoldRenderOptions.aovMode",  1)
        cmds.setAttr("defaultArnoldDriver.aiTranslator",    "exr", type="string")
        cmds.setAttr("defaultArnoldDriver.mergeAOVs",       1)

    def _render_view(self, view_name, bb_min, bb_max):
        self._frame_camera_tetrahedral(view_name, bb_min, bb_max)
        self._set_render_globals_exr()

        exr_prefix = os.path.join(self.export_path, view_name)
        cmds.setAttr("defaultArnoldDriver.prefix", exr_prefix, type="string")

        print(f"  [render] {view_name} merged AOV EXR ...")
        cmds.arnoldRender(width=self.resolution, height=self.resolution,
                          camera=self.cam_shape, batch=True)

        return {"exr": os.path.join(self.export_path, f"{view_name}.exr")}

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------

    def run(self):
        self.ensure_dir(self.export_path)
        self._init_arnold()
        self._setup_aovs()

        transforms         = self.get_transforms()
        bb_min, bb_max     = self.get_bounding_box(transforms)

        self._create_lights(bb_min, bb_max)
        self._override_shader()
        self._create_ortho_cam()

        all_passes = {}
        # Iterate over the 4 tetrahedral face views
        for view_name in self._tet_views:
            print(f"\n[view] === {view_name.upper()} ===")
            all_passes[view_name] = self._render_view(view_name, bb_min, bb_max)

        # Cleanup
        if cmds.objExists(self.cam_name):
            cmds.delete(self.cam_name)
        self._delete_lights()

        print("\n=== GeometryPlanarExtractor DONE ===")
        print(f"  Output folder : {self.export_path}")
        print(f"  Views rendered: {list(all_passes.keys())}")
        return all_passes
