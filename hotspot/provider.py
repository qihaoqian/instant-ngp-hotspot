import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
import pysdf

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def plot_results(results):
    """
    Visualize surface points and free space points simultaneously:
      - Surface points use results['points_surf'] and results['sdfs_surf']
      - Free space points use results['points_free'] and results['sdfs_free']
    The map_color function is used to map SDF values to colors.
    """
    # Extract data
    points_surf = results['points_surf']  # [N, 3]
    sdfs_surf = results['sdfs_surf']     # [N, 1]
    points_free = results['points_free'] # [M, 3]
    sdfs_free = results['sdfs_free']     # [M, 1]

    # Map colors (assuming the map_color function is already defined)
    colors_surf = map_color(sdfs_surf.squeeze(1))
    colors_free = map_color(sdfs_free.squeeze(1))

    # Construct trimesh point cloud objects
    pc_surf = trimesh.PointCloud(points_surf, colors_surf)
    pc_free = trimesh.PointCloud(points_free, colors_free)

    # Add both point clouds to the same scene for simultaneous observation
    scene = trimesh.Scene([pc_surf, pc_free])
    scene.show()   

def plot_results_separately(results):
    """
    Visualize surface points and free space points separately using trimesh.

    Parameters:
      results: dict with keys 'points_surf', 'sdfs_surf', 'points_free', 'sdfs_free'.
    """
    import trimesh

    # Extract data for surface points
    points_surf = results['points_surf']  # [N, 3]
    sdfs_surf = results['sdfs_surf']        # [N, 1]
    
    # Extract data for free space points
    points_free = results['points_free']    # [M, 3]
    sdfs_free = results['sdfs_free']          # [M, 1]

    # Map SDF values to colors (assuming map_color is defined elsewhere)
    colors_surf = map_color(sdfs_surf.squeeze(1))
    colors_free = map_color(sdfs_free.squeeze(1))

    # Construct trimesh point cloud objects
    pc_surf = trimesh.PointCloud(points_surf, colors_surf)
    pc_free = trimesh.PointCloud(points_free, colors_free)

    # Create separate scenes for each point cloud
    scene_surf = trimesh.Scene([pc_surf])
    scene_free = trimesh.Scene([pc_free])

    # Display the scenes independently
    scene_surf.show(title="Surface Points")
    scene_free.show(title="Free Space Points")


# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        # online sampling
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        points_surf = self.mesh.sample(self.num_samples * 7 // 8)
        # perturb surface
        points_surf[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        
        sdfs_surf = np.zeros((self.num_samples * 7 // 8, 1))
        sdfs_surf[self.num_samples // 2:] = -self.sdf_fn(points_surf[self.num_samples // 2:])[:,None]
        # random
        points_free = np.random.rand(self.num_samples // 8, 3) * 2 - 1

        sdfs_free = -self.sdf_fn(points_free)[:,None]
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs_surf': sdfs_surf.astype(np.float32),
            'points_surf': points_surf.astype(np.float32),
            'sdfs_free': sdfs_free.astype(np.float32),
            'points_free': points_free.astype(np.float32),
        }

        plot_results_separately(results)

        return results
