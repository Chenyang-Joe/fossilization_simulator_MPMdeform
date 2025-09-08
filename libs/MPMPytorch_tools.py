import torch
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation


def select_particles_bulk(particles, point, normal):
    point = torch.tensor(point, device=particles.device, dtype=particles.dtype)
    normal = torch.tensor(normal, device=particles.device, dtype=particles.dtype)

    diff = particles - point  # shape (N, 3)
    dot = torch.matmul(diff, normal)  # (N,)
    
    return dot < 0  # left side of the plane (negative dot product)



def lighten_faces(mesh, disk_radius, disk_center, projection_direction):
    # return the barycenter of the faces
    face_barycenters = mesh.triangles_center  # (num_faces, 3)

    rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)  # fast (needs pyembree)

    lighten_idx = []
    dist_list = []
    for i, face_center in enumerate(face_barycenters):
        connection = face_center - disk_center
        # length of the projection of connection onto projection_direction
        distance = np.abs(np.linalg.norm(np.cross(connection, projection_direction))/np.linalg.norm(projection_direction))  # perpendicular distance
        dist_list.append(distance)
        if distance < disk_radius:
            locations, index_ray, index_tri = rmi.intersects_location(
                ray_origins=face_center.reshape(1, 3),
                ray_directions= -1 * projection_direction.reshape(1, 3),
                multiple_hits=True
            )
            projected = False
            if len(locations) == 0: 
                projected = True
            elif len(locations) == 1 and np.linalg.norm(locations - face_center) < 1e-6:
                projected = True
            if projected:          
                lighten_idx.append(i)

    return lighten_idx


def points_covered_by_voxel(pc, vg):
    N = len(pc)
    
    # World -> voxel index coordinates
    Tinvt = np.linalg.inv(vg.transform)        # 4x4
    hom = np.c_[pc, np.ones(N)]                 # (N,4)
    ijk_f = (Tinvt @ hom.T).T[:, :3]           # float index coords
    ijk = np.floor(ijk_f + 1e-9).astype(int)   # containing cell (i,j,k)
    
    # Inside-grid mask
    sx, sy, sz = vg.shape  # dims along (i,j,k)
    valid = (
        (ijk[:,0] >= 0) & (ijk[:,0] < sx) &
        (ijk[:,1] >= 0) & (ijk[:,1] < sy) &
        (ijk[:,2] >= 0) & (ijk[:,2] < sz)
    )

    inside_occ = np.zeros(N, dtype=bool)  # default False outside grid
    if np.any(valid):
        occ = vg.matrix[ijk[valid,0], ijk[valid,1], ijk[valid,2]]  # bool
        inside_occ[valid] = occ
    
    # point indices that land in occupied voxels:
    pt_idx = np.nonzero(inside_occ)[0]
    # their voxel indices:
    # vox_ijk = ijk[inside_occ]
    

    return pt_idx


def random_unit_on_circle(normal):
    """
    normal: (3,) array-like, disk axis
    returns: (3,) unit vector lying in the plane perpendicular to `normal`,
             uniformly random on the unit circle.
    """
    n = np.asarray(normal, float)
    ln = np.linalg.norm(n)
    if ln == 0:
        raise ValueError("normal must be non-zero")
    n /= ln

    # Orthonormal basis (u, v) spanning plane ⟂ n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)  # unit, ⟂ n and ⟂ u

    # Random angle
    theta = 2 * np.pi * np.random.random()
    return np.cos(theta) * u + np.sin(theta) * v


def select_particles_surface(PC, Mesh, direction, radius, disk_shift, grid_res):
    # Define disk center
    box = Mesh.bounding_box
    box_center = box.centroid
    box_extents = box.extents
    center_distance = 2 * np.linalg.norm(box_extents)/2
    random_disk_circle_direction = random_unit_on_circle(direction)
    disk_center =  box_center + center_distance * direction  + disk_shift *np.random.random() * random_disk_circle_direction

    # Find projected faces
    lighten_faces_idx = lighten_faces(Mesh, radius, disk_center, -1 * direction)

    # Find covered points
    projected_mesh = Mesh.submesh([lighten_faces_idx], only_watertight=False, append=True)
    if not isinstance(projected_mesh, trimesh.Trimesh):
        # No faces selected, handle gracefully
        return np.array([], dtype=int)
    surface_res = 2 * grid_res
    pitch = 1/surface_res
    vg = projected_mesh.voxelized(pitch)          # only surface voxel
    surf_occ = vg.matrix.astype(bool)   # (Z,Y,X)
    T = vg.transform   

    surf_occ = binary_dilation(surf_occ, structure=np.ones((3,3,3), bool), iterations=2)
    vg_thick = trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(surf_occ), transform=vg.transform)

    pt_idx = points_covered_by_voxel(PC , vg_thick)

    # eval_mesh = trimesh.PointCloud(PC[pt_idx])
    # eval_mesh.export(f"dev/eval_{direction[0]}_{direction[1]}_{direction[2]}_{radius}".replace(".","_")+".obj")
    # return selection
    return pt_idx