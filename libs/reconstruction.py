import numpy as np
import time
from scipy.spatial import cKDTree


def find_closest(v, mesh, num, rng=0.1):
    # boolean mask: keep any point where **any** coord is within ±rng
    mask = np.any(np.abs(mesh - v) <= rng, axis=1)


    # if no one passes, use the whole mesh
    if not mask.any():
        mask[:] = True


    # subset and compute distances
    cand = mesh[mask]
    dists = np.linalg.norm(cand - v, axis=1)


    # pick k smallest (unordered)
    local = np.argpartition(dists, num)[:num]


    # map back to original indices
    orig_idxs = np.flatnonzero(mask)[local]


    return orig_idxs, dists[local]




def build_mesh_tree(mesh):
    return cKDTree(mesh)    




def find_closest_tree(v, mesh, tree, num, rng=0.1):
    # Prefilter by Chebyshev ball (max‐norm ≤ rng)
    idxs_box = tree.query_ball_point(v, rng, p=np.inf)


    if idxs_box:
        # we have some candidates — compute true Euclidean dists
        pts = mesh[idxs_box]
        d = np.linalg.norm(pts - v, axis=1)
        # pick the k smallest among those (unordered)
        # local = np.argpartition(d, num)[:num]
        m = d.size
        if m <= num:
            # fewer points than requested → just sort them all
            local = np.argsort(d)
        else:    
            # plenty of points → get the num smallest (unordered)
            local = np.argpartition(d, num)[:num]


        return np.array(idxs_box)[local], d[local]
    else:
        # fallback to a pure k‐NN query
        dists, idxs = tree.query(v, k=num)
        # handle k=1 case (scalar→1-elem array)
        if num == 1:
            dists = np.array([dists])
            idxs  = np.array([idxs])
        return idxs, dists




def deform_point(v, dist_list, points_ref_before, points_ref_after, method = "IDW"):
    epsilon = 1e-10
    weights = 1.0 / (dist_list**2 + epsilon)      # shape (N,)
    weights /= weights.sum()                     # normalize
    raw_transform = points_ref_after - points_ref_before                  # shape (N, D)
    v_transform = weights.dot(raw_transform)     # shape (D,)


    return v + v_transform




def mesh_relation(low_mesh, high_mesh, num = 100, rng=0.1):
    # num: the max number of points to search for each point in low_mesh
    # rng: the range to search for each point in low_mesh
    idx_results = []
    distances_results = []
    mesh_tree = build_mesh_tree(high_mesh)
    for i, v in enumerate(low_mesh):
        idx, distances = find_closest_tree(v, high_mesh, mesh_tree, num, rng)
        idx_results.append(idx)
        distances_results.append(distances)
        if i % 1000 ==0:
            print(f"Process the {i}/{low_mesh.shape[0]} vertex in low mesh")


    return idx_results, distances_results


def restore_mesh(low_mesh, idx_results, distances_results, high_mesh_before, high_mesh_after, method = "IDW"):
    new_low_mesh = np.zeros(low_mesh.shape)
    for i, v in enumerate(low_mesh):
        dist_list = distances_results[i]
        points_ref_before = high_mesh_before[idx_results[i]]
        points_ref_after = high_mesh_after[idx_results[i]]
        new_low_mesh[i] = deform_point(v, dist_list, points_ref_before, points_ref_after, method = method)
        if i % 10000 ==0:
            print(f"{i}/{low_mesh.shape[0]}")


    return new_low_mesh


