import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import pysdf
from hotspot.utils import plot_sdf_slice
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, size=100, 
                 num_samples_surf=20000, 
                 num_samples_space=10000,
                 clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')
        self.pq = trimesh.proximity.ProximityQuery(self.mesh)

        # # normalize to [-1, 1] (different from instant-sdf where it is [0, 1])
        # vs = self.mesh.vertices
        # vmin = vs.min(0)
        # vmax = vs.max(0)
        # v_center = (vmin + vmax) / 2
        # v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        # vs = (vs - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        
        # Compute bounds of the mesh and scale it to fit within a sphere
        min_bound = self.mesh.bounds[0]  # Minimum corner of the bounding box
        max_bound = self.mesh.bounds[1]  # Maximum corner of the bounding box 
        center = (min_bound + max_bound) / 2
        bbox_diagonal = np.linalg.norm(max_bound - min_bound)  # Diagonal length of the bounding box
        scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)  # Scale so that the object's diagonal fits within the sphere
        
        self.mesh.vertices -= center  # Translate mesh to the origin
        self.mesh.vertices *= scale_factor  # Scale the mesh

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples_surf = num_samples_surf
        self.num_samples_space = num_samples_space

        self.size = size


    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # Surface sampling
        points_surf = self.mesh.sample(self.num_samples_surf)
        
        # Perturbation of the surface is commented out
        # points_surf[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 4, 3)
        # sdfs_surf = np.zeros((self.num_samples * 3 // 4, 1))
        # sdfs_surf[self.num_samples // 2:] = -self.sdf_fn(points_surf[self.num_samples // 2:])[:, None]
        
        sdfs_surf = np.zeros((self.num_samples_surf, 1))
        
        # Randomly sample points in space and compute their corresponding SDF values
        points_space = np.random.rand(self.num_samples_space, 3) * 2 - 1   # shape: (N, 3)
        
        # # Uniform and Gaussian random sampling
        # # First sample from a Gaussian distribution centered at 0, then clip to [-1,1]
        # N = self.num_samples_space
        # alpha = 0.5   # 70% using Gaussian, 30% using uniform
        # n_gauss = int(alpha * N)
        # n_uniform = N - n_gauss
        # 
        # sigma = 0.4   # mean 0, standard deviation 0.4
        # g = np.clip(np.random.randn(n_gauss, 3) * sigma, -1, 1)
        # u = np.random.rand(n_uniform, 3) * 2 - 1
        # points_space = np.vstack([g, u])
        
        sdfs_space = -self.sdf_fn(points_space)[:, None]  # shape: (N, 1)

        # Construct mask: points with SDF < 0 are considered "occupied", otherwise "free"
        mask = (sdfs_space[:, 0] < 0)  # shape: (N,)

        # Extract points for both classes
        points_occupied = points_space[mask]  # points inside the object
        points_free = points_space[~mask]       # points outside the object

        sdfs_occupied = sdfs_space[mask]
        sdfs_free = sdfs_space[~mask]

        results = {
            'sdfs_surf': sdfs_surf.astype(np.float32),
            'points_surf': points_surf.astype(np.float32),
            'sdfs_occupied': sdfs_occupied.astype(np.float32),
            'points_occupied': points_occupied.astype(np.float32),
            'sdfs_free': sdfs_free.astype(np.float32),
            'points_free': points_free.astype(np.float32),
            # 'points_space_dists': dists.astype(np.float32),
        }

        return results
    
    def plot_dataset_sdf_slice(self,
                       workspace: str = None,
                       axis: str = 'z',
                       coord: float = 0.0,
                       resolution: int = 256,
                       val_range: tuple = (-1.0, 1.0),
                       cmap: str = 'jet'):
        """
        Slice the SDF dataset along a given axis.
        
        This function takes a slice of the space at the specified coordinate (e.g., y=0) along the chosen axis,
        constructs a resolution x resolution grid in the other two dimensions, computes the ground-truth SDF for each point,
        and displays a heat map.
        
        Parameters:
          axis: Axis along which to take the slice. Options are 'x', 'y', or 'z'.
          coord: The coordinate on the slicing axis where the slice is taken.
          resolution: Grid resolution for the slice.
          val_range: Sampling range (min, max). The same range is used for the other two dimensions.
          cmap: Colormap name from matplotlib.
        """
        # Generate a 2D grid
        v0, v1 = val_range
        grid = np.linspace(v0, v1, resolution)
        A, B = np.meshgrid(grid, grid, indexing='xy')  # shape: (res, res)

        # Expand the grid to a 3D point set based on the slicing axis
        if axis == 'x':
            pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
        elif axis == 'y':
            pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
        elif axis == 'z':
            pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
        else:
            raise ValueError(f"Unsupported axis {axis}")

        pts_flat = pts.reshape(-1, 3).astype(np.float64)  # (res², 3)

        # Call pysdf to compute the ground-truth SDF. Note that the sign convention depends on your setup.
        sdf_flat = -self.sdf_fn(pts_flat)  # shape: (res²,)
        sdf_slice = sdf_flat.reshape(resolution, resolution)

        dims = ['x', 'y', 'z']
        dims.remove(axis)   # e.g., for axis='y', dims -> ['x', 'z']

        xlabel, ylabel = dims  # First dimension as x-axis, second as y-axis

        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            sdf_slice.T,
            origin='lower',
            extent=[v0, v1, v0, v1],
            cmap=cmap,
            vmin=np.min(sdf_slice),
            vmax=np.max(sdf_slice)
        )
        plt.colorbar(im, label='SDF value')
        plt.xlabel(xlabel)  # e.g., 'x'
        plt.ylabel(ylabel)  # e.g., 'z'
        plt.title(f"SDF slice on {axis}={coord:.2f}")
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_slice_dataset.png", dpi=300)
        # plt.show()