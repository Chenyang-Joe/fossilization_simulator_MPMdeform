import torch

def select_particles(particles, point, normal):
    point = torch.tensor(point, device=particles.device, dtype=particles.dtype)
    normal = torch.tensor(normal, device=particles.device, dtype=particles.dtype)

    diff = particles - point  # shape (N, 3)
    dot = torch.matmul(diff, normal)  # (N,)
    
    return dot < 0  # left side of the plane (negative dot product)