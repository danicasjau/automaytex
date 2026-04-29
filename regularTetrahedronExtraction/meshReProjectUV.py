import os
import math
import re

try:
    import maya.cmds as cmds
    import maya.api.OpenMaya as om
    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False

from PIL import Image


class UVRetargetTool:
    def __init__(self, mesh, output_dir=None, config=None):
        if config is None:
            from config import configuration          # type: ignore[import]
            config = configuration()
        self.config = config
        self.mesh   = mesh
        self.output_dir = output_dir or self.config.output_path

        os.makedirs(self.output_dir, exist_ok=True)

        # Populated by getOriginalUV()
        self.original_uvs: dict  = {}
        self.original_uv_set: str | None = None

        # Populated by setMaterialTextures()
        self.texturesPaths: list[str] = []

    # ------------------------------------------------------------------
    # PUBLIC – STORE ORIGINAL UVs
    # ------------------------------------------------------------------

    def getOriginalUV(self, uv_set: str | None = None) -> None:
        if not MAYA_AVAILABLE:
            raise RuntimeError("Maya is not available in this environment.")

        sel = om.MSelectionList()
        sel.add(self.mesh)
        dag = sel.getDagPath(0)
        
        try:
            dag.extendToShape()
        except:
            pass

        fn  = om.MFnMesh(dag)

        if not uv_set:
            uv_set = fn.currentUVSetName() or (fn.getUVSetNames() or ["map1"])[0]

        self.original_uv_set = uv_set

        try:
            u_array, v_array     = fn.getUVs(uv_set)
            uv_counts, uv_ids   = fn.getAssignedUVs(uv_set)
        except Exception as exc:
            print(f"[WARNING] Could not read UV set '{uv_set}': {exc}")
            u_array = v_array = uv_counts = uv_ids = []

        self.original_uvs = {
            "u":        list(u_array),
            "v":        list(v_array),
            "uvCounts": list(uv_counts),
            "uvIds":    list(uv_ids),
            "uv_set":   uv_set,
        }
        print(f"[INFO] Stored UV set '{uv_set}' "
              f"({len(u_array)} UV points, {len(uv_counts)} faces).")

    # ------------------------------------------------------------------
    # PUBLIC – SET SOURCE TEXTURES
    # ------------------------------------------------------------------

    def setMaterialTextures(self, texturesPaths: list[str]) -> None:
        self.texturesPaths = list(texturesPaths)

    # ------------------------------------------------------------------
    # PUBLIC – FIND UV SETS
    # ------------------------------------------------------------------
    
    def _get_target_uv_set(self, fn: om.MFnMesh) -> str:
        """Finds the primary default UV set to project the bake onto."""
        if self.original_uv_set:
            return self.original_uv_set
        uv_sets = fn.getUVSetNames()
        return "map1" if "map1" in uv_sets else uv_sets[0]

    # ------------------------------------------------------------------
    # PUBLIC – CREATE TETRAHEDRAL PLANAR UV
    # ------------------------------------------------------------------

    def createTetrahedralPlanarUV(self) -> None:
        """
        Creates the 'planarUV' set by projecting the mesh faces onto the 4 planes
        of a regular tetrahedron, matching the extraction cameras.
        This must be called before retargetToOriginalUV() if planarUV does not exist.
        """
        if not MAYA_AVAILABLE:
            raise RuntimeError("Maya is not available.")
        
        sel = om.MSelectionList()
        sel.add(self.mesh)
        dag = sel.getDagPath(0)
        
        # Bounding box
        bb_min = [math.inf]*3
        bb_max = [-math.inf]*3
        bb = cmds.exactWorldBoundingBox(self.mesh)
        for i in range(3):
            bb_min[i] = bb[i]
            bb_max[i] = bb[i+3]
            
        cx = (bb_min[0] + bb_max[0]) * 0.5
        cy = (bb_min[1] + bb_max[1]) * 0.5
        cz = (bb_min[2] + bb_max[2]) * 0.5
        
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        dz = bb_max[2] - bb_min[2]
        bb_radius = 0.5 * math.sqrt(dx*dx + dy*dy + dz*dz)
        if bb_radius == 0.0: bb_radius = 1.0
        ortho_w = bb_radius * 2.0 * (1.0 + 0.1)

        s2 = math.sqrt(2)
        s23 = math.sqrt(2.0 / 3.0)
        tet_normals = {
            "face_0": ( 0.0,            -1.0,       0.0   ),
            "face_1": (-2*s2/3,          1.0/3,     0.0   ),
            "face_2": ( s2/3,            1.0/3,    -s23   ),
            "face_3": ( s2/3,            1.0/3,     s23   ),
        }

        # Normalize tet_normals
        for k in tet_normals:
            nx, ny, nz = tet_normals[k]
            l = math.sqrt(nx*nx + ny*ny + nz*nz)
            tet_normals[k] = (nx/l, ny/l, nz/l)

        # Classify faces
        try:
            dag.extendToShape()
        except:
            pass
        it = om.MItMeshPolygon(dag)
        prefix = dag.fullPathName() + ".f["
        groups = {k: [] for k in tet_normals}

        while not it.isDone():
            n = it.getNormal(om.MSpace.kWorld)
            best_face = None
            max_dot = -math.inf
            for k, (nx, ny, nz) in tet_normals.items():
                dot = n.x * nx + n.y * ny + n.z * nz
                if dot > max_dot:
                    max_dot = dot
                    best_face = k
            
            # Avoid degenerate faces
            length_sq = n.x*n.x + n.y*n.y + n.z*n.z
            if length_sq > 1e-10 and best_face is not None:
                groups[best_face].append(prefix + str(it.index()) + "]")
            it.next()

        if "planarUV" not in cmds.polyUVSet(self.mesh, q=True, allUVSets=True) or []:
            cmds.polyUVSet(self.mesh, create=True, uvSet="planarUV")
        cmds.polyUVSet(self.mesh, currentUVSet=True, uvSet="planarUV")

        face_order = ["face_0", "face_1", "face_2", "face_3"]
        for idx, view_name in enumerate(face_order):
            faces = groups[view_name]
            if not faces:
                continue

            nx, ny, nz = tet_normals[view_name]
            rx = math.degrees(-math.asin(max(-1.0, min(1.0, ny))))
            ry = math.degrees(math.atan2(nx, nz))
            rz = 0.0

            cmds.polyPlanarProjection(
                faces,
                rx=rx, ry=ry, rz=rz,
                pc=(cx, cy, cz),
                pw=ortho_w, ph=ortho_w,
                icx=0.5, icy=0.5,
                md="p",
                name="planarUV"
            )

            col = idx % 2
            row = idx // 2
            
            chunk_size = 1000
            for i in range(0, len(faces), chunk_size):
                cmds.polyEditUV(faces[i:i+chunk_size], uValue=col, vValue=row)

        print("[INFO] Created planarUV set for 4 tetrahedral planes.")

    # ------------------------------------------------------------------
    # PUBLIC – MAIN RETARGET
    # ------------------------------------------------------------------

    def retargetToOriginalUV(self, resolution: int = 1024) -> None:
        if not MAYA_AVAILABLE:
            raise RuntimeError("Maya is not available.")

        if not self.texturesPaths:
            print("[WARNING] No textures registered – call setMaterialTextures() first.")
            return

        sel = om.MSelectionList()
        sel.add(self.mesh)
        dag = sel.getDagPath(0)

        # Ensure we are looking at the shape node for UV sets
        try:
            dag.extendToShape()
        except:
            pass

        fn  = om.MFnMesh(dag)

        # ── 1. Prepare UV sets ─────────────────────────────────────────
        src_uv_set = "planarUV"
        dst_uv_set = self._get_target_uv_set(fn)
        
        if src_uv_set not in fn.getUVSetNames():
            print(f"[WARNING] Mesh '{self.mesh}' is missing '{src_uv_set}'.")
            return

        # ── 2. Load source UDIM tiles ──────────────────────────────────
        src_images: dict[int, Image.Image] = {}
        for tex_path in self.texturesPaths:
            for udim_str, path in self._detect_udims(tex_path).items():
                try:
                    src_images[int(udim_str)] = Image.open(path).convert("RGBA")
                    print(f"[INFO] Loaded source tile UDIM {udim_str}: {path}")
                except Exception as exc:
                    print(f"[WARNING] Could not load '{path}': {exc}")

        if not src_images:
            print("[WARNING] No valid source UDIM tiles loaded.")
            return

        # ── 3. Allocate output buffers (created on demand) ─────────────
        dst_images: dict[int, Image.Image] = {}

        # ── 4. Rasterise every face ────────────────────────────────────
        it          = om.MItMeshPolygon(dag)
        total_faces = it.count()
        face_idx    = 0

        print(f"[INFO] Rasterising {total_faces} faces at {resolution}px …")

        while not it.isDone():
            if face_idx % 500 == 0:
                print(f"  … face {face_idx}/{total_faces}")
            face_idx += 1

            try:
                u_src, v_src = it.getUVs(src_uv_set)
                u_dst, v_dst = it.getUVs(dst_uv_set)
            except Exception:
                it.next()
                continue

            n = len(u_src)
            if n < 3 or len(u_dst) < 3:
                it.next()
                continue

            # Fan-triangulate the polygon (works for convex polys)
            for i in range(n - 2):
                tri_src = [
                    (u_src[0],   v_src[0]),
                    (u_src[i+1], v_src[i+1]),
                    (u_src[i+2], v_src[i+2]),
                ]
                tri_dst = [
                    (u_dst[0],   v_dst[0]),
                    (u_dst[i+1], v_dst[i+1]),
                    (u_dst[i+2], v_dst[i+2]),
                ]
                self._rasterize_triangle(
                    tri_dst, tri_src,
                    src_images, dst_images,
                    resolution,
                )
            it.next()

        # ── 5. Dilate each tile to fill seam gaps ─────────────────────
        for udim, img in dst_images.items():
            dst_images[udim] = self._dilate(img, passes=2)

        # ── 6. Save output tiles ───────────────────────────────────────
        for udim, img in dst_images.items():
            out_path = os.path.join(self.output_dir, f"retarget_{udim}.png")
            # Flip vertically before saving
            final_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            final_img.save(out_path)
            print(f"[INFO] Saved vertically flipped image → {out_path}")

        # ── 7. Clean up Temporary Planar UVs ──────────────────
        try:
            cmds.polyUVSet(self.mesh, currentUVSet=True, uvSet=dst_uv_set)
            if "planarUV" in fn.getUVSetNames():
                cmds.polyUVSet(self.mesh, delete=True, uvSet="planarUV")
            print(f"[INFO] Retarget complete. Cleared planarUV. Original '{dst_uv_set}' is cleanly untouched!")
        except Exception as e:
            print(f"[WARNING] Native UV cleanup failed: {e}")

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    # ------------------------------------------------------------------
    # Rasterise one triangle
    # ------------------------------------------------------------------

    def _rasterize_triangle(
        self,
        tri_dst: list[tuple[float, float]],
        tri_src: list[tuple[float, float]],
        src_images: dict[int, Image.Image],
        dst_images: dict[int, Image.Image],
        resolution: int,
    ) -> None:

        # Bounding box in UV space (may span multiple UDIM tiles)
        min_u = min(p[0] for p in tri_dst)
        max_u = max(p[0] for p in tri_dst)
        min_v = min(p[1] for p in tri_dst)
        max_v = max(p[1] for p in tri_dst)

        # Determine which UDIM tiles this triangle can touch
        tile_u_min = int(math.floor(min_u))
        tile_u_max = int(math.floor(max_u))
        tile_v_min = int(math.floor(min_v))
        tile_v_max = int(math.floor(max_v))

        for tile_v in range(tile_v_min, tile_v_max + 1):
            for tile_u in range(tile_u_min, tile_u_max + 1):
                udim = 1001 + tile_u + tile_v * 10

                # Pixel bounding box clipped to this tile
                px_min = max(0, int(math.floor((min_u - tile_u) * resolution)))
                px_max = min(resolution - 1,
                             int(math.ceil( (max_u - tile_u) * resolution)))
                py_min = max(0, int(math.floor((min_v - tile_v) * resolution)))
                py_max = min(resolution - 1,
                             int(math.ceil( (max_v - tile_v) * resolution)))

                for px in range(px_min, px_max + 1):
                    for py in range(py_min, py_max + 1):
                        # Pixel centre → global UV1 coordinate
                        u = tile_u + (px + 0.5) / resolution
                        v = tile_v + (py + 0.5) / resolution

                        bc = self._barycentric(
                            (u, v), tri_dst[0], tri_dst[1], tri_dst[2]
                        )
                        if bc is None:
                            continue

                        # Interpolate UV2 coordinate
                        u_src = (bc[0] * tri_src[0][0]
                                 + bc[1] * tri_src[1][0]
                                 + bc[2] * tri_src[2][0])
                        v_src = (bc[0] * tri_src[0][1]
                                 + bc[1] * tri_src[1][1]
                                 + bc[2] * tri_src[2][1])

                        color = self._sample_bilinear(u_src, v_src, src_images)
                        if color is None or color[3] == 0:
                            continue

                        # Ensure output tile exists
                        if udim not in dst_images:
                            dst_images[udim] = Image.new(
                                "RGBA", (resolution, resolution), (0, 0, 0, 0)
                            )
                        dst_images[udim].putpixel((px, py), color)

    # ------------------------------------------------------------------
    # Bilinear texture sampling (replaces nearest-neighbour getpixel)
    # ------------------------------------------------------------------

    def _sample_bilinear(
        self,
        u: float,
        v: float,
        src_images: dict[int, Image.Image],
    ) -> tuple | None:
        """
        Bilinear sample at UV coordinate *(u, v)* from the UDIM atlas.

        Returns ``None`` when the UDIM tile is not loaded.
        """
        tile_u = int(math.floor(u))
        tile_v = int(math.floor(v))
        udim   = 1001 + tile_u + tile_v * 10

        img = src_images.get(udim)
        if img is None:
            return None

        w, h      = img.size
        local_u   = u - tile_u
        local_v   = v - tile_v

        # Map to pixel space with Y-flip (Maya V=0 is Bottom, Image Y=0 is Top)
        fx = local_u * w - 0.5
        fy = (1.0 - local_v) * h - 0.5

        x0 = max(0, int(math.floor(fx)))
        y0 = max(0, int(math.floor(fy)))
        x1 = min(w - 1, x0 + 1)
        y1 = min(h - 1, y0 + 1)

        tx = fx - math.floor(fx)
        ty = fy - math.floor(fy)

        c00 = img.getpixel((x0, y0))
        c10 = img.getpixel((x1, y0))
        c01 = img.getpixel((x0, y1))
        c11 = img.getpixel((x1, y1))

        def lerp_chan(a, b, t):
            return a + (b - a) * t

        result = tuple(
            int(
                lerp_chan(
                    lerp_chan(c00[ch], c10[ch], tx),
                    lerp_chan(c01[ch], c11[ch], tx),
                    ty,
                )
            )
            for ch in range(4)
        )
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Barycentric coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _barycentric(
        p:  tuple[float, float],
        a:  tuple[float, float],
        b:  tuple[float, float],
        c:  tuple[float, float],
    ) -> tuple[float, float, float] | None:
        """
        Return (w1, w2, w3) such that ``p = w1·a + w2·b + w3·c``,
        or ``None`` when:
          • The triangle is degenerate (area ≈ 0), or
          • ``p`` lies outside the triangle.
        """
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-10:
            return None  # degenerate / zero-area triangle

        w1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
        w2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
        w3 = 1.0 - w1 - w2

        if w1 < -1e-6 or w2 < -1e-6 or w3 < -1e-6:
            return None  # outside triangle

        return (w1, w2, w3)

    # ------------------------------------------------------------------
    # Dilation (seam gap fill)
    # ------------------------------------------------------------------

    @staticmethod
    def _dilate(img: Image.Image, passes: int = 2) -> Image.Image:
        """
        Expand opaque pixels into transparent neighbours to cover seam gaps.

        A simple box-dilation: each transparent pixel takes the average colour
        of its opaque 4-connected neighbours.  Repeated ``passes`` times.
        """
        import numpy as np

        arr = np.array(img, dtype=np.float32)  # H×W×4
        for _ in range(passes):
            alpha   = arr[:, :, 3]
            opaque  = alpha > 0

            filled  = arr.copy()
            h, w    = arr.shape[:2]

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rolled  = np.roll(arr,   shift=dy, axis=0)
                rolled  = np.roll(rolled, shift=dx, axis=1)
                r_opq   = np.roll(opaque, shift=dy, axis=0)
                r_opq   = np.roll(r_opq,  shift=dx, axis=1)

                mask = (~opaque) & r_opq   # transparent dst, opaque neighbour
                filled[mask] = rolled[mask]

            arr = filled

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    # ------------------------------------------------------------------
    # UDIM detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_udims(path: str) -> dict[str, str]:
        import re
        import os
        
        # If no <UDIM> token, check if the file itself has a UDIM standard index embedded
        if "<UDIM>" not in path:
            match = re.search(r'10\d{2}', os.path.basename(path))
            if match:
                return {match.group(): path}
            return {"1001": path}

        # Build a regex from the template, escaping everything except <UDIM>
        folder   = os.path.dirname(path)
        basename = os.path.basename(path)

        # Replace <UDIM> with a capturing group matching 4-digit UDIM numbers
        pattern = re.escape(basename).replace(r"\<UDIM\>", r"(1[0-9]{3})")
        regex   = re.compile(r"^" + pattern + r"$")

        udims: dict[str, str] = {}
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                m = regex.match(fname)
                if m:
                    udims[m.group(1)] = os.path.join(folder, fname)

        if not udims:
            print(f"[WARNING] No UDIM tiles matched template: {path}")

        return udims