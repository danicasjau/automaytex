import maya.cmds as cmds
import maya.api.OpenMaya as om
import math

# ---------------------------------------------------------------------------
# GeoPlanarUVProjection  – optimised, large-mesh safe
# Preserves: plane directions, pivot position, UDIM 1001-1006 assignment
# ---------------------------------------------------------------------------

class GeoPlanarUVProjection:
    def __init__(self, output_dir=None, config=None):
        self.config = config
        self.output_dir = output_dir

    # ------------------------------------------------------------------ utils


    @staticmethod
    def _world_bbox(transforms):
        """Return (min[3], max[3]) world bounding box across all transforms."""
        mn = [ math.inf,  math.inf,  math.inf]
        mx = [-math.inf, -math.inf, -math.inf]
        for tfm in transforms:
            bb = cmds.exactWorldBoundingBox(tfm)   # xmin ymin zmin xmax ymax zmax
            for i in range(3):
                if bb[i]     < mn[i]: mn[i] = bb[i]
                if bb[i + 3] > mx[i]: mx[i] = bb[i + 3]
        return mn, mx

    @staticmethod
    def _chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # ------------------------------------------------- face classification

    def _classify_faces(self, dag_path):
        """
        Single-pass face classification using MItMeshPolygon.
        Returns dict {view_name: [face_component_strings]}.
        Skips zero-length normals (degenerate / lamina faces).
        """
        groups = {v: [] for v in self.config.view_rotations}

        it = om.MItMeshPolygon(dag_path)
        # Build prefix once – avoids per-iteration attribute look-up
        prefix = dag_path.fullPathName() + ".f["

        while not it.isDone():
            n = it.getNormal(om.MSpace.kWorld)
            nx, ny, nz = n.x, n.y, n.z
            ax, ay, az = abs(nx), abs(ny), abs(nz)

            # Skip degenerate faces (zero-area polygons produce zero normals)
            length_sq = ax * ax + ay * ay + az * az
            if length_sq < 1e-10:
                it.next()
                continue

            face_str = prefix + str(it.index()) + "]"

            if ax >= ay and ax >= az:
                groups["left" if nx > 0 else "right"].append(face_str)
            elif ay >= ax and ay >= az:
                groups["top"  if ny > 0 else "bottom"].append(face_str)
            else:
                groups["front" if nz > 0 else "back"].append(face_str)

            it.next()

        return groups

    # ------------------------------------------------- projection per mesh

    def _project_mesh(self, tfm, cx, cy, cz, dx, dy, dz, padding):
        """Apply 6-planar projection + UDIM shift for a single transform."""

        sel = om.MSelectionList()
        try:
            sel.add(tfm)
        except Exception as e:
            cmds.warning(f"[GeoPlanarUV] Could not add '{tfm}' to selection: {e}")
            return

        dag = sel.getDagPath(0)

        # Extend dag to its mesh shape so MItMeshPolygon works correctly
        dag.extendToShape()

        groups = self._classify_faces(dag)

        for view_name in self.config.face_order_6:
            faces = groups[view_name]
            if not faces:
                continue

            # Projection dimensions – identical logic to original
            if view_name in ("front", "back"):
                horiz, vert = dx, dy
            elif view_name in ("left", "right"):
                horiz, vert = dz, dy
            else:
                horiz, vert = dx, dz

            ortho_w = max(horiz, vert) * (1.0 + padding)
            rx, ry, rz = self.config.view_rotations[view_name]

            # Create and switch to planarUV safely
            if "planarUV" not in cmds.polyUVSet(tfm, q=True, allUVSets=True) or []:
                cmds.polyUVSet(tfm, create=True, uvSet="planarUV")
            cmds.polyUVSet(tfm, currentUVSet=True, uvSet="planarUV")

            # Single projection call for all faces of this view
            cmds.polyPlanarProjection(
                faces,
                rx=rx, ry=ry, rz=rz,
                pc=(cx, cy, cz),
                pw=ortho_w, ph=ortho_w,
                icx=0.5, icy=0.5,
                md="p",
                name="planarUV" # Explicitly force the command to write here
            )

            # UDIM shift (3x2 grid for 6 faces, 2x2 grid for 4 faces)
            idx      = self.config.face_order_6.index(view_name)
            num_faces = len(self.config.face_order_6)
            cols      = 3 if num_faces == 6 else 2
            
            col      = idx % cols
            row      = idx // cols
            u_shift  = float(col)
            v_shift  = float(row)

            # Chunked polyEditUV – avoids stack overflow on high poly meshes
            for batch in self._chunk(faces, self.config.uv_chunk_size):
                cmds.polyEditUV(batch, uValue=u_shift, vValue=v_shift)

            om.MGlobal.displayInfo(
                f"[GeoPlanarUV]  {tfm}  '{view_name}'  →  UDIM {1001 + idx}"
                f"  ({len(faces)} faces)"
            )

    # ------------------------------------------------------------------ run

    def run(self, selection=None):
        sel = selection if selection is not None else cmds.ls(sl=True, long=True)

        # Resolve to transforms, following shape parents if needed
        transforms = cmds.ls(sel, type="transform", long=True) or []
        if not transforms:
            shapes = cmds.ls(sel, dag=True, shapes=True, long=True) or []
            if shapes:
                parents = {
                    cmds.listRelatives(s, parent=True, fullPath=True)[0]
                    for s in shapes
                }
                transforms = list(parents)

        if not transforms:
            cmds.warning("[GeoPlanarUV] No geometry found. Select at least one mesh.")
            return

        # One bbox pass across all meshes
        bb_min, bb_max = self._world_bbox(transforms)
        cx = (bb_min[0] + bb_max[0]) * 0.5
        cy = (bb_min[1] + bb_max[1]) * 0.5
        cz = (bb_min[2] + bb_max[2]) * 0.5
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        dz = bb_max[2] - bb_min[2]

        padding = self.config.ortho_padding

        total = len(transforms)
        for i, tfm in enumerate(transforms, 1):
            om.MGlobal.displayInfo(
                f"[GeoPlanarUV] Processing {i}/{total}: {tfm}"
            )
            self._project_mesh(tfm, cx, cy, cz, dx, dy, dz, padding)

        om.MGlobal.displayInfo(
            "=== GeoPlanarUVProjection DONE === "
            f"UDIMs 1001-1006  |  order: {self.config.face_order_6}"
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    GeoPlanarUVProjection().run()