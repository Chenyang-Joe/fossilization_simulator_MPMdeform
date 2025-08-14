import json
import trimesh
import open3d as o3d
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector


class SimulatorDeform:
    def __init__(self, config_json):
        """
        Initialize the simulator with a JSON configuration file.
        """
        self.config_json = config_json
        self.model_path = config_json["model_path"]
        self.faces_limitation = int(config_json.get("faces_limitation", None))
        self.mesh = None 
        self.MPM_before = None
        self.MPM_after = None
        self.grid_res = int(config_json["grid_res"])


    def load_raw_mesh(self):
        print("Loading raw mesh")
        self.mesh = trimesh.load(self.model_path)
        print(f"#Vertices: {len(self.mesh.vertices)}")
        print(f"#Faces: {len(self.mesh.faces)}")


    def downsampling(self):
        # downsampling if self.faces_limitation set
        if self.faces_limitation and (self.faces_limitation <= len(self.mesh.faces)):
            print(f"Downsampling to #Faces {self.faces_limitation}")
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.mesh.vertices),
                o3d.utility.Vector3iVector(self.mesh.faces)
            )
            simplified = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=self.faces_limitation)
            self.mesh = trimesh.Trimesh(
                vertices=np.asarray(simplified.vertices),
                faces=np.asarray(simplified.triangles)
            )
        else:
            print("Do not need to downsampling.")

    def rescale(self):
        print("Rescaling")
        vs = np.asarray(self.mesh.vertices)
        vs -= vs.min(axis=0)
        vs /= vs.max()
        self.mesh.vertices = vs

    def pruning(self, min_faces = 10, deep_pruning = True):
        print("Pruning")
        # delete all components with too few faces, so all inner points are deleted.
        components = self.mesh.split(only_watertight=False)
        print(f"#Connected components: {len(components)}")
        filtered = [comp for comp in components if len(comp.faces) >= min_faces]
        if len(filtered) > 0:
            self.mesh = trimesh.util.concatenate(filtered)
            print(f"Keep {len(filtered)} components")
        else:
            raise ValueError("All components have been filtered!")

        if deep_pruning:
            # Other typical pruning steps
            self.mesh.remove_duplicate_faces()
            self.mesh.remove_degenerate_faces()
            self.mesh.remove_unreferenced_vertices()

    def preprocess(self):
        print("Start preprocess")
        self.load_raw_mesh()
        self.downsampling()
        self.pruning()
        self.rescale()
        print("Done preprocess\n")


    def generate_MPM(self):
        print("Generate MPM before deformation")
        # from surface mesh to MPM point cloud
        x = np.linspace(0, 1, self.grid_res)
        y = np.linspace(0, 1, self.grid_res)
        z = np.linspace(0, 1, self.grid_res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        self.mesh.ray = RayMeshIntersector(self.mesh)
        if isinstance(self.mesh.ray, RayMeshIntersector):
            print("Pyembree acceleration turned on")
        else:
            print("Pyembree acceleartion did not turn on")

        inside = self.mesh.contains(grid_points)
        points_inside = grid_points[inside]
        self.MPM_before = trimesh.points.PointCloud(points_inside)
        self.MPM_before.export("mpm_before_example.obj")
        print(f"Points contained:{len(points_inside)}/{inside.shape[0]}\n")



