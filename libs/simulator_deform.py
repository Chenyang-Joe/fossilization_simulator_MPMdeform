import json
import trimesh
import open3d as o3d
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector
import torch
from torch import Tensor
from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive
from functools import partial
from .MPMPytorch_tools import *
import tqdm


class SimulatorDeform:
    def __init__(self, config_json):
        """
        Initialize the simulator with a JSON configuration file.
        """
        self.config_json = config_json
        self.model_path = config_json["model_path"]
        self.faces_limitation = int(config_json.get("faces_limitation", None))
        self.mesh = None 
        self.PC_before = None
        self.PC_after = None
        self.grid_res = int(config_json["grid_res"])
        self.MPMPytorch_config = config_json["MPMPytorch_config"]


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


    def generate_PC(self):
        print("Generate the points cloud before deformation")
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
        self.PC_before = trimesh.points.PointCloud(points_inside)
        print(f"Points contained:{len(points_inside)}/{inside.shape[0]}\n")


    def MPMPytorch_init(self):
        print("Initing MPMPytorch config")
        self.device = self.MPMPytorch_config.get("device", None)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Pytorch uses device: {self.device}")
        self.num_pre_particle_process_function = 0
        self.n_particles = 0
        self.deform_config_list = []
        center_list = self.MPMPytorch_config["center"].split(" ")
        self.center = [float(center_list[0]),
                       float(center_list[1]),
                       float(center_list[2])]
        self.scale_factor = self.MPMPytorch_config["scale_factor"]
        self.elasticity_type = self.MPMPytorch_config["elasticity_type"]
        self.plasticity_type = self.MPMPytorch_config["plasticity_type"]
        init_velocity_list = self.MPMPytorch_config["initial_velocity"].split(" ")
        self.initial_velocity = [float(init_velocity_list[0]),
                                 float(init_velocity_list[1]),
                                 float(init_velocity_list[2])]
        self.num_frames = self.MPMPytorch_config["num_frames"]
        self.steps_per_frame = self.MPMPytorch_config["steps_per_frame"]
        self.frames = []


    def to_pytorch_PC(self):
        print("Converting pc to pytorch format")
        vs_PC = self.PC_before.vertices
        self.pytorch_PC = torch.tensor(vs_PC, dtype=torch.float32)
        center_tensor = torch.tensor(self.center, dtype=torch.float32)
        self.pytorch_PC = (self.pytorch_PC - center_tensor) * self.scale_factor + center_tensor
        self.pytorch_PC = self.pytorch_PC.to(self.device)
        self.n_particles = self.pytorch_PC.shape[0]



    def export_MPMPytorch_PC(self, MPMPytorch_PC, save_path="./MPMPytorch_PC_example.obj"):
        MPMPytorch_PC_save = MPMPytorch_PC.clone()
        vs = MPMPytorch_PC_save.cpu().numpy()
        PC = trimesh.points.PointCloud(vs)
        PC.export(save_path)
        print(f"Save PC example to {save_path}")

    def setup_MPMSolver(self):
        print("Setting up MPMSolver")
        self.mpm_solver = MPMSolver(
            self.pytorch_PC, 
            enable_train=False,
            device=self.device,
            gravity=[0.0, 0.0, 0.0]
        )
        # Boundary condition
        self.elasticity = get_constitutive(self.elasticity_type, device=self.device)
        self.plasticity = get_constitutive(self.plasticity_type, device=self.device)

    def add_pre_particle_process_function(self, deform_config, start_time = 0, end_time=999):
        select:Tensor = deform_config["select"]
        factor:float = deform_config["factor"]
        direction:list[float] = deform_config["direction"]
        def my_deform(model: MPMSolver, x: Tensor, v:Tensor, start_time: float, end_time:float, select: Tensor):

            time = model.time
            unit_displacement = torch.tensor(direction, device=model.device).float()
            unit_displacement = unit_displacement / unit_displacement.norm()
            displacement = 0.000001 * factor
            if time >= start_time and time < end_time:
                # x[select] = x[select] + unit_displacement * displacement
                v[select] = v[select] + unit_displacement * displacement/model.dt
        
        self.mpm_solver.pre_particle_process.append(
            partial(
                my_deform,
                start_time=start_time,
                end_time=end_time,
                select = select
            )
        )
        self.num_pre_particle_process_function += 1
        print("Added a new pre_particle_process_function")
        print(f"The num of pre_particle_process_function is {self.num_pre_particle_process_function}")


    def load_deform_config(self):
        print("Load deformation config")
        # analysis model cordinate stats
        coords_static = self.pytorch_PC.cpu().numpy()  # Extract x-coordinates

        x_max = np.max(coords_static[:,0])
        x_min = np.min(coords_static[:,0])
        y_max = np.max(coords_static[:,1])
        y_min = np.min(coords_static[:,1])
        z_max = np.max(coords_static[:,2])
        z_min = np.min(coords_static[:,2])
        print(f"x from {x_min} to {x_max}")
        print(f"y from {y_min} to {y_max}")
        print(f"z from {z_min} to {z_max}")


        # Another function should generate these info randomly.
        # This function should switch to load these info generated.
        # Also evaluate vis for output.
        point1 = [0.3, 0.4, 0.4]
        direction1 = [1, 0, 0.5]
        factor1 = 1
        point2 = [0.4, 0.4, 0.4]
        direction2 = [0, -1, 0.]
        factor2 = 1
        point3 = [0.4, 0.4, 0.6]
        direction3 = [0, 0.3, -1]
        factor3 = 1
        select1 = select_particles(self.pytorch_PC, point1, direction1)
        select2 = select_particles(self.pytorch_PC, point2, direction2)
        select3 = select_particles(self.pytorch_PC, point3, direction3)
        self.deform_config_list.append({"select":select1
                                        ,"factor":factor1
                                        ,"direction":direction1})
        self.deform_config_list.append({"select":select2
                                        ,"factor":factor2
                                        ,"direction":direction2})
        self.deform_config_list.append({"select":select3
                                        ,"factor":factor3
                                        ,"direction":direction3})

        for deform_config in self.deform_config_list:
            self.add_pre_particle_process_function(deform_config)

    def run_simulation(self):
        x = self.pytorch_PC
        v = torch.stack([torch.tensor(self.initial_velocity, device=self.device) for _ in range(self.n_particles)])
        C = torch.zeros((self.n_particles, 3, 3), device=self.device)
        F = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.n_particles, 1, 1)

        for frame in tqdm.tqdm(range(self.num_frames), desc='Simulating'):
            self.frames.append(x.cpu().numpy())
            for step in tqdm.tqdm(range(self.steps_per_frame), desc='Step'):
                # Update stress
                stress = self.elasticity(F)
                # Particle to grid, grid update, grid to particle
                x, v, C, F = self.mpm_solver(x, v, C, F, stress)
                # Plasticity correction
                F = self.plasticity(F)
        self.frames.append(x.cpu().numpy())

        self.export_MPMPytorch_PC(x)

        PC_result = x.clone()
        v = PC_result.cpu().numpy()
        self.PC_after = trimesh.points.PointCloud(v)


    def MPMPytorch_deform(self):
        self.MPMPytorch_init()
        self.to_pytorch_PC()
        self.setup_MPMSolver()
        self.load_deform_config()
        self.run_simulation()

