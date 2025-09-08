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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .reconstruction import *
import math
import gc
import torch
import os


class SimulatorDeformFragments:
    def __init__(self, config_json):
        """
        Initialize the simulator with a JSON configuration file.
        """
        self.config_json = config_json
        self.model_path = config_json["model_path"]
        self.save_folder = config_json["save_folder"]
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.faces_limitation = int(config_json.get("faces_limitation", None))
        self.vanilla_mesh_list = [] 
        self.model_path_list = []
        self.vanilla_mesh_concat = None
        self.final_mesh_list = []
        self.PC_before = None
        self.PC_after = None
        self.grid_res = int(config_json["grid_res"])
        self.MPMPytorch_config = config_json["MPMPytorch_config"]


    def load_raw_mesh(self):
        print("Loading raw mesh")
        for model in os.listdir(self.model_path):
            if model.endswith('.obj') or model.endswith('.ply') or model.endswith('.stl'):
                model_path_full = os.path.join(self.model_path, model)
                print(f"Loading {model_path_full}")
                current_mesh = trimesh.load(model_path_full)
                self.vanilla_mesh_list.append(current_mesh)
                self.model_path_list.append(model_path_full)
        self.vanilla_mesh_concat = trimesh.util.concatenate(self.vanilla_mesh_list)
        print(f"#Vertices: {len(self.vanilla_mesh_concat.vertices)}")
        print(f"#Faces: {len(self.vanilla_mesh_concat.faces)}")

    def downsampling(self):
        # downsampling if self.faces_limitation set
        if self.faces_limitation and (self.faces_limitation <= len(self.vanilla_mesh_concat.faces)):
            print(f"Downsampling to #Faces {self.faces_limitation}")
            for i, mesh in enumerate(self.vanilla_mesh_list):
                target_faces = int(self.faces_limitation * len(mesh.faces) / len(self.vanilla_mesh_concat.faces))
                o3d_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(mesh.vertices),
                    o3d.utility.Vector3iVector(mesh.faces)
                )
                simplified = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                self.vanilla_mesh_list[i] = trimesh.Trimesh(
                    vertices=np.asarray(simplified.vertices),
                    faces=np.asarray(simplified.triangles)
                )
        else:
            print("Do not need to downsampling.")

    def rescale(self):
        print("Rescaling")
        vs = np.asarray(self.vanilla_mesh_concat.vertices)
        vs_min = vs.min(axis=0)
        vs_range = vs.max()
        vs -= vs_min
        vs /= vs_range
        self.vanilla_mesh_concat.vertices = vs

        for i, mesh in enumerate(self.vanilla_mesh_list):
            mesh.vertices = (mesh.vertices - vs_min) / vs_range
            self.vanilla_mesh_list[i] = mesh

    def pruning(self, min_faces = 10, deep_pruning = True):
        print("Pruning")
        # delete all components with too few faces, so all inner points are deleted.
        for i, mesh in enumerate(self.vanilla_mesh_list):
            components = mesh.split(only_watertight=False)
            print(f"#Connected components: {len(components)}")
            filtered = [comp for comp in components if len(comp.faces) >= min_faces]
            if len(filtered) > 0:
                mesh = trimesh.util.concatenate(filtered)
                print(f"Keep {len(filtered)} components")
            else:
                print("All components have been filtered!")
                # raise ValueError("All components have been filtered!")
            if deep_pruning:
                # Other typical pruning steps
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                mesh.remove_unreferenced_vertices()
            self.vanilla_mesh_list[i] = mesh
        self.vanilla_mesh_concat = trimesh.util.concatenate(self.vanilla_mesh_list)

    def preprocess(self):
        print("Start preprocess")
        self.load_raw_mesh()
        self.downsampling()
        self.pruning()
        self.rescale()
        self.vanilla_mesh_concat.export("./data/vanilla_mesh.obj")
        print("Done preprocess\n")


    def generate_PC(self):
        print("Generate the points cloud before deformation")
        # from surface mesh to MPM point cloud
        x = np.linspace(0, 1, self.grid_res)
        y = np.linspace(0, 1, self.grid_res)
        z = np.linspace(0, 1, self.grid_res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        self.vanilla_mesh_concat.ray = RayMeshIntersector(self.vanilla_mesh_concat)
        if isinstance(self.vanilla_mesh_concat.ray, RayMeshIntersector):
            print("Pyembree acceleration turned on")
        else:
            print("Pyembree acceleartion did not turn on")

        inside = self.vanilla_mesh_concat.contains(grid_points)
        points_inside = grid_points[inside]
        self.PC_before = trimesh.points.PointCloud(points_inside)
        print(f"Points contained:{len(points_inside)}/{inside.shape[0]}\n")

######################todo########################### --- IGNORE ---

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


    def PC_to_pytorch_PC(self):
        print("Converting pc to pytorch format")
        vs_PC = self.PC_before.vertices
        self.pytorch_PC = torch.tensor(vs_PC, dtype=torch.float32)
        center_tensor = torch.tensor(self.center, dtype=torch.float32)
        self.pytorch_PC = (self.pytorch_PC - center_tensor) * self.scale_factor + center_tensor
        self.pytorch_PC = self.pytorch_PC.to(self.device)
        self.n_particles = self.pytorch_PC.shape[0]

    def pytorch_PC_to_PC(self):
        center_tensor = torch.tensor(self.center, dtype=torch.float32, device = self.device)
        self.MPMPC_result = (self.MPMPC_result - center_tensor) / self.scale_factor + center_tensor
        v = self.MPMPC_result.cpu().numpy()
        self.PC_after = trimesh.points.PointCloud(v)



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
        # print("Added a new pre_particle_process_function")
        # print(f"The num of pre_particle_process_function is {self.num_pre_particle_process_function}")

    def generate_random_direction(self): # use np.random
        # This function generates a random unit vector in 3D space.
        # It uses spherical coordinates to ensure uniform distribution over the sphere.
        # Because dA = \sin\theta \, d\theta \, d\phi = dz \times d\phi, which is uniformly sampled by changing \z and \phi.
        z = 2.0 * np.random.random() - 1.0          # uniform in [-1, 1]
        phi = 2.0 * math.pi * np.random.random()    # uniform in [0, 2π)
        r = math.sqrt(1.0 - z*z)
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        
        return np.array([x, y, z])               # already unit length

    def select_region(self):
        pc_before_pytorch = self.pytorch_PC.clone()
        pc_before = pc_before_pytorch.cpu().numpy()
        mesh_rescaled = self.vanilla_mesh_concat.copy()
        mesh_rescaled.vertices = (mesh_rescaled.vertices - self.center) * self.scale_factor + self.center

        force_config = self.MPMPytorch_config["force_info"]

        # major force
        major_force_config = force_config["major_force"]
        force_direction_major = major_force_config.get("region_proj_direction", None) # pointing from the object center
        if not force_direction_major:
            force_direction_major = self.generate_random_direction()
        else:
            force_direction_major = force_direction_major/np.linalg.norm(force_direction_major)
        print("Major force direction: ", force_direction_major)
        radius_major = major_force_config.get("radius", 0.2)
        force_magnitude_major = major_force_config.get("force_magnitude", 2)
        max_shift_major = major_force_config.get("disk_shift", 0.1)
        selection_major = select_particles_surface(
            pc_before,
            mesh_rescaled,
            force_direction_major,
            radius_major,
            max_shift_major,
            self.grid_res
            ) 
        self.deform_config_list.append({"select":selection_major
                                        ,"factor":force_magnitude_major
                                        ,"direction":-1 * force_direction_major})

        # minor force
        minor_force_configs = force_config.get("minor_forces", None)
        if minor_force_configs:
            for minor_force_config in minor_force_configs:
                radius_minor = minor_force_config.get("radius", 0.1)
                force_magnitude_minor = minor_force_config.get("force_magnitude", 0.5)
                num_minor = minor_force_config.get("num", 1)
                max_shift_minor = minor_force_config.get("disk_shift", 0.1)
                for i in range(num_minor):
                    force_direction_minor = self.generate_random_direction()
                    selection_minor= select_particles_surface(
                    pc_before,
                    mesh_rescaled,
                    force_direction_minor,
                    radius_minor,
                    max_shift_minor,
                    self.grid_res
                    ) 
                    self.deform_config_list.append({"select":selection_minor
                                                    ,"factor":force_magnitude_minor
                                                    ,"direction":-1 * force_direction_minor})


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


        # Manually select points, bulkly selection
        # point1 = [0.3, 0.4, 0.4]
        # direction1 = [1, 0, 0.5]
        # factor1 = 1
        # point2 = [0.4, 0.4, 0.4]
        # direction2 = [0, -1, 0.]
        # factor2 = 1
        # point3 = [0.4, 0.4, 0.6]
        # direction3 = [0, 0.3, -1]
        # factor3 = 1
        # select1 = select_particles_bulk(self.pytorch_PC, point1, direction1)
        # select2 = select_particles_bulk(self.pytorch_PC, point2, direction2)
        # select3 = select_particles_bulk(self.pytorch_PC, point3, direction3)
        # self.export_MPMPytorch_PC(self.pytorch_PC, save_path="./dev/MPMPC_before.obj")
        # self.deform_config_list.append({"select":select1
        #                                 ,"factor":factor1
        #                                 ,"direction":direction1})
        # self.deform_config_list.append({"select":select2
        #                                 ,"factor":factor2
        #                                 ,"direction":direction2})
        # self.deform_config_list.append({"select":select3
        #                                 ,"factor":factor3
        #                                 ,"direction":direction3})

        self.select_region()




        for deform_config in self.deform_config_list:
            self.add_pre_particle_process_function(deform_config)

        animate_config = self.MPMPytorch_config.get("animate_info", False)
        if animate_config:
            print("Animation is enabled")
            self.animate = True
            self.gif_save_dir = animate_config.get("gif_save_dir", self.model_path.split(".")[0] + "MPM_animate.gif")
            self.sample_rate = animate_config.get("sample_rate", 30)
        else:
            self.animate = False
            print("Animation is disabled")



    def save_animation(self):
        new_frames = self.frames[::self.sample_rate]

        s = 20
        c = 'blue'
        fps = 10

        print(f"Rendering to {self.gif_save_dir}...")
        size = [1, 1, 1]
        xlim = [self.center[0] - size[0] / 2, self.center[0] + size[0] / 2]
        ylim = [self.center[1] - size[1] / 2, self.center[1] + size[1] / 2]
        zlim = [self.center[2] - size[2] / 2, self.center[2] + size[2] / 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter([], [], [], s=s)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)  
        ax.set_zlim(zlim)
        def update(frame_idx):
            print(f"updating the {frame_idx}/{len(new_frames)} frame")
            ax.cla()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            scat = ax.scatter(new_frames[frame_idx][:, 0], new_frames[frame_idx][:, 1], new_frames[frame_idx][:, 2], s=s, c=c)
            ax.set_title(f"Frame {frame_idx * self.sample_rate}")
            return scat
        ani = FuncAnimation(fig, update, frames=len(new_frames), blit=False)
        ani.save(self.gif_save_dir, writer='pillow', fps=fps)
        plt.close()


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

        self.MPMPC_result = x.clone()
        # self.export_MPMPytorch_PC(self.MPMPC_result, save_path="./dev/MPMPC_after.obj")


        if self.animate:
            self.save_animation()

    def MPMPytorch_deform(self):
        self.MPMPytorch_init()
        self.PC_to_pytorch_PC()
        self.setup_MPMSolver()
        self.load_deform_config()
        self.run_simulation()
        self.pytorch_PC_to_PC()

    def Mesh_reconstruction(self):
        print("Start mesh reconstruction")
        reconstruction_config = self.config_json.get("reconstruction_config", None)
        if reconstruction_config:
            method = reconstruction_config.get("method", "IDW")
            num_neighbors = reconstruction_config.get("num_neighbors", 100)
            search_radius = reconstruction_config.get("search_radius", 0.2)
        else:
            method = "IDW"
            num_neighbors = 100
            search_radius = 0.2


        print("Convert point cloud to mesh")
        v_before = np.asarray(self.PC_before.vertices)
        v_after = np.asarray(self.PC_after.vertices)
        v_vanilla_list = []
        for mesh in self.vanilla_mesh_list:
            v_vanilla_list.append(np.asarray(mesh.vertices))

        print("Mesh bounding box extents: ", self.vanilla_mesh_concat.bounding_box.extents)
        print("Point Cloud before deformation bounding box extents: ", self.PC_before.bounding_box.extents)
        print("Point Cloud after deformation bounding box extents: ",self.PC_after.bounding_box.extents)

        self.final_mesh_list = []
        for i, v_vanilla in enumerate(v_vanilla_list):
            idx_results, distances_results = mesh_relation(v_vanilla, v_before, num=num_neighbors, rng=search_radius)
            v_deformed = restore_mesh(v_vanilla, idx_results, distances_results, v_before, v_after, method = method)
            self.final_mesh_list.append(trimesh.Trimesh(vertices=v_deformed, faces=self.vanilla_mesh_list[i].faces))
            save_name = os.path.join(self.save_folder, self.model_path_list[i].split("/")[-1].split(".")[0] + "_deformed.obj")
            self.final_mesh_list[i].export(save_name)
            print(f"Save deformed mesh to {save_name}")
        self.final_mesh_concat = trimesh.util.concatenate(self.final_mesh_list)
        self.final_mesh_concat.export(self.save_folder + "deformed_result_concat.obj")
        print("Done mesh reconstruction\n")
    
    def clean_up(self):
        """
        Free up GPU memory and clear references to large objects.
        """
        attrs = [
            'pytorch_PC', 'MPMPC_result', 'mpm_solver', 'elasticity', 'plasticity', 'frames', 'final_mesh', 'PC_before', 'PC_after', 'vanilla_mesh'
        ]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
        gc.collect()

