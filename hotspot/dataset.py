import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import pysdf
from hotspot.utils import plot_sdf_slice


# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        # vs = self.mesh.vertices
        # vmin = vs.min(0)
        # vmax = vs.max(0)
        # v_center = (vmin + vmax) / 2
        # v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        # vs = (vs - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        
        min_bound = self.mesh.bounds[0]  # Min corner of the bounding box
        max_bound = self.mesh.bounds[1]  # Max corner of the bounding box 
        center = (min_bound + max_bound) / 2
        bbox_diagonal = np.linalg.norm(max_bound - min_bound)  # Diagonal length of the bounding box
        scale_factor = 0.5 * np.sqrt(3) / (bbox_diagonal / 2)  # Scale so that the object's diagonal fits within the sphere
        
        self.mesh.vertices -= center  # Translate mesh to the origin
        self.mesh.vertices *= scale_factor  # Scale the mesh

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # surface
        points_surf = self.mesh.sample(self.num_samples * 7 // 8)
        
        # perturb surface
        points_surf[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        sdfs_surf = np.zeros((self.num_samples * 7 // 8, 1))
        sdfs_surf[self.num_samples // 2:] = -self.sdf_fn(points_surf[self.num_samples // 2:])[:, None]
        
        # free space
        points_free = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        sdfs_free = -self.sdf_fn(points_free)[:, None]

        results = {
            'sdfs_surf': sdfs_surf.astype(np.float32),
            'points_surf': points_surf.astype(np.float32),
            'sdfs_free': sdfs_free.astype(np.float32),
            'points_free': points_free.astype(np.float32),
        }

        # ## Visualize combined points
        # points = np.concatenate([points_surf, points_free], axis=0)
        # sdfs = np.concatenate([sdfs_surf, sdfs_free], axis=0)
        
        # plot_sdf_slice(points, sdfs)

        return results

    # def __getitem__(self, _):
    #     # surface
    #     points_surf = self.mesh.sample(self.num_samples *3 // 4)
        
    #     # # perturb surface
    #     # points_surf[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
    #     sdfs_surf = np.zeros((self.num_samples *3 // 4, 1))
    #     # sdfs_surf[self.num_samples // 2:] = -self.sdf_fn(points_surf[self.num_samples // 2:])[:, None]
        
    #     # free space
    #     points_free = np.random.rand(self.num_samples // 4, 3) * 2 - 1
    #     sdfs_free = -self.sdf_fn(points_free)[:, None]

    #     results = {
    #         'sdfs_surf': sdfs_surf.astype(np.float32),
    #         'points_surf': points_surf.astype(np.float32),
    #         'sdfs_free': sdfs_free.astype(np.float32),
    #         'points_free': points_free.astype(np.float32),
    #     }

    #     ## Visualize combined points
    #     # points = np.concatenate([points_surf, points_free], axis=0)
    #     # sdfs = np.concatenate([sdfs_surf, sdfs_free], axis=0)
        
    #     # plot_sdf_slice(points, sdfs)

    #     return results