import os
import cv2
import numpy as np

class mapsMaterialGenerator():
    def __init__(self, imagesToGenerate, diffuseImagePath, normalImagePath, outputPath=None):
        self.initial_normalCollageImagePath = normalImagePath

        self.diffuseImagePath = diffuseImagePath

        self.output_path = outputPath

        self.prompt = "detailed material texture, highly detailed, 8k"
        self.tex_generator = None

        self.imagesToGenerate = imagesToGenerate
    
    def create(self):
        if not self.output_path:
            self.output_path = "./texturesOutput"

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        print("[Info] Starting material maps creation...")

        if "diffuse" in self.imagesToGenerate:
            self.diffuseImagePath = self.create_diffuseMap(self.diffuseImagePath)
        else:
            self.diffuseImagePath = None
        
        if "roughness" in self.imagesToGenerate:
            self.roughnessImagePath = self.create_roughnessMap(self.diffuseImagePath)
        else:
            self.roughnessImagePath = None

        if "metalness" in self.imagesToGenerate:
            self.metalnessImagePath = self.create_metalnessMap(self.diffuseImagePath)
        else:
            self.metalnessImagePath = None

        if "height" in self.imagesToGenerate:
            self.heightImagePath = self.create_heightMap(self.diffuseImagePath)
        else:
            self.heightImagePath = None

        if "normal" in self.imagesToGenerate:
            self.normalImagePath = self.create_normalMap(self.heightImagePath)
        else:
            self.normalImagePath = None
        
        print("[Info] Material maps created successfully in:", self.output_path)



        return {
            "diffuse": self.diffuseImagePath,
            "roughness": self.roughnessImagePath,
            "metalness": self.metalnessImagePath,
            "height": self.heightImagePath,
            "normal": self.normalImagePath
        }
    
    def create_diffuseMap(self, image_path):
        if False:
            if not self.tex_generator:
                self.tex_generator = mPiplineCreationt.TextureGenerator()
                
            out_file = os.path.join(self.output_path, "diffuse.png")
            print(f"[Info] Generating Diffuse Map: {out_file}")
            
            self.tex_generator.generate_texture(
                prompt=self.prompt,
                input_image_path=image_path,
                output_path=out_file
            )
            return out_file
        else:
            diffuse_path = cv2.imread(image_path)
            cv2.imwrite(os.path.join(self.output_path, "diffuse.png"), diffuse_path)
            self.diffuseImagePath = image_path
            return image_path

    def create_roughnessMap(self, image_path):
        out_file = os.path.join(self.output_path, "roughness.png")
        print(f"[Info] Generating Roughness Map: {out_file}")
        
        diffuse = cv2.imread(image_path)
        if diffuse is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        gray = cv2.cvtColor(diffuse, cv2.COLOR_BGR2GRAY)
        roughness = cv2.bitwise_not(gray) # Intuitively, darker colors can be rougher, flipped here
        # Soften to avoid extreme 0/1 values
        roughness = cv2.addWeighted(roughness, 0.7, np.full_like(roughness, 128), 0.3, 0)
        
        cv2.imwrite(out_file, roughness)

        self.roughnessImagePath = out_file

        return out_file

    def create_metalnessMap(self, image_path):
        out_file = os.path.join(self.output_path, "metalness.png")
        print(f"[Info] Generating Metalness Map: {out_file}")
        
        diffuse = cv2.imread(image_path)
        if diffuse is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        gray = cv2.cvtColor(diffuse, cv2.COLOR_BGR2GRAY)
        # Assuming most generated surfaces are dielectric by default
        metalness = np.zeros_like(gray)
        cv2.imwrite(out_file, metalness)

        self.metalnessImagePath = out_file

        return out_file

    def create_normalMap(self, image_path):
        out_file = os.path.join(self.output_path, "normal.png")
        print(f"[Info] Generating Normal Map: {out_file}")
        
        height = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if height is None:
            raise ValueError(f"Could not read height image: {image_path}")

        generator = PBRMapGenerator()
        detail_normals = generator.height_to_normal(height)
        
        # --- Apply Gaussian Blur to Detail Normals ---
        # (5, 5) is the kernel size; 0 lets OpenCV calculate sigma based on kernel size.
        detail_normals = cv2.GaussianBlur(detail_normals, (5, 5), 0)
        
        if self.initial_normalCollageImagePath and os.path.exists(self.initial_normalCollageImagePath):
            base_normals = cv2.imread(self.initial_normalCollageImagePath)
            base_normals_rgb = cv2.cvtColor(base_normals, cv2.COLOR_BGR2RGB)
            
            BLUR_SIZE = 99
            # --- Apply Gaussian Blur to Base Normals ---
            base_normals_rgb = cv2.GaussianBlur(base_normals_rgb, (BLUR_SIZE, BLUR_SIZE), 0)
            
            target_shape = (detail_normals.shape[1], detail_normals.shape[0])
            if base_normals_rgb.shape[:2] != detail_normals.shape[:2]:
                base_normals_rgb = cv2.resize(base_normals_rgb, target_shape, interpolation=cv2.INTER_LINEAR)
            
            final_normals_rgb = generator.blend_rnm(base_normals_rgb, detail_normals)
            final_normals = cv2.cvtColor(final_normals_rgb, cv2.COLOR_RGB2BGR)
        else:
            final_normals = cv2.cvtColor(detail_normals, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(out_file, final_normals)
        self.normalImagePath = out_file
        return out_file

    def create_heightMap(self, image_path):
        out_file = os.path.join(self.output_path, "height.png")
        print(f"[Info] Generating Height Map: {out_file}")
        
        diffuse = cv2.imread(image_path)
        if diffuse is None:
            raise ValueError(f"Could not read diffuse image: {image_path}")

        rough_path = os.path.join(self.output_path, "roughness.png")
        roughness = cv2.imread(rough_path, cv2.IMREAD_GRAYSCALE)
        if roughness is None:
            raise ValueError(f"Could not read roughness image: {rough_path}")

        metal_path = os.path.join(self.output_path, "metalness.png")
        metalness = cv2.imread(metal_path, cv2.IMREAD_GRAYSCALE)
        if metalness is None:
            raise ValueError(f"Could not read metalness image: {metal_path}")
        
        generator = PBRMapGenerator()
        height = generator.generate_height_map(diffuse, roughness, metalness)
        cv2.imwrite(out_file, height)
        self.heightImagePath = out_file
        return out_file

    def setOutputPath(self, output_path):
        self.output_path = output_path

    def getFiles(self):
        return {
            "diffuse": self.diffuseImagePath,
            "roughness": self.roughnessImagePath,
            "metalness": self.metalnessImagePath,
            "height": self.heightImagePath,
            "normal": self.normalImagePath
        }


class PBRMapGenerator:
    def __init__(self, strength=1.0, detail_boost=1.0):
        self.strength = strength
        self.detail_boost = detail_boost

    def generate_height_map(self, diffuse, roughness, metalness):
        """
        Creates a heuristic height map by combining PBR channels.
        Darker diffuse and rougher areas are usually 'lower'.
        """
        if diffuse is None or roughness is None or metalness is None:
            raise ValueError("Input maps to generate_height_map cannot be None")

        # Convert diffuse to grayscale for luminance
        gray_diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        rough = roughness.astype(np.float32) / 255.0
        metal = metalness.astype(np.float32) / 255.0

        # Heuristic: Height is often inverse of roughness and influenced by albedo
        # We use a weighted sum to prioritize roughness for surface structure
        height = (gray_diffuse * 0.3) + ((1.0 - rough) * 0.7)
        
        # Metalness adjustment: metals are often smoother/flatter
        height = cv2.addWeighted(height, 0.9, (1.0 - metal), 0.1, 0)
        
        return np.clip(height * 255, 0, 255).astype(np.uint8)

    def height_to_normal(self, height_map):
        """
        Converts a height map to a tangent space normal map using Sobel filters.
        """
        height_float = height_map.astype(np.float32) / 255.0
        
        # Calculate gradients
        dx = cv2.Sobel(height_float, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(height_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Assemble normal vector
        # X = dx, Y = dy, Z = 1/strength
        ones = np.ones_like(height_float)
        normal = np.stack([dx * self.strength, dy * self.strength, ones], axis=2)
        
        # Normalize vectors to unit length
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / norm
        
        # Map from [-1, 1] to [0, 255]
        normal = (normal * 0.5 + 0.5) * 255
        return normal.astype(np.uint8)


    def blend_rnm(self, base_norm, detail_norm):
        """
        Blends two normal maps using Reoriented Normal Mapping (RNM).
        base_norm: The rendered normals from your low-poly geometry.
        detail_norm: The 'fake' normals from the diffuse/roughness.
        """
        # Unpack to [-1, 1]
        n1 = (base_norm.astype(np.float32) / 255.0) * 2.0 - 1.0
        n2 = (detail_norm.astype(np.float32) / 255.0) * 2.0 - 1.0

        # RNM Math
        z1 = n1[:, :, 2:3] + 1.0
        n1_xy = n1[:, :, :2]
        n2_xy = n2[:, :, :2]
        n2_z = n2[:, :, 2:3]

        # Reorientation step
        r = n1_xy * (n2_z / z1) - n2_xy
        
        # Reconstruct and Normalize
        combined = np.concatenate([r, n2_z], axis=2)
        mag = np.linalg.norm(combined, axis=2, keepdims=True)
        combined = combined / mag

        # Back to [0, 255]
        return ((combined * 0.5 + 0.5) * 255).astype(np.uint8)


    def process_all(self, low_mesh_normals, diffuse, roughness, metalness):
        height = self.generate_height_map(diffuse, roughness, metalness)
        detail_normals = self.height_to_normal(height)
        final_normals = self.blend_rnm(low_mesh_normals, detail_normals)
        
        return height, final_normals

