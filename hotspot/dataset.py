import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import pysdf
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

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces, robust=False)
        
        self.num_samples_surf = num_samples_surf
        self.num_samples_space = num_samples_space

        self.size = size


    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # Surface sampling
        points_surf = self.mesh.sample(self.num_samples_surf)
        
        # Perturbation of the surface is commented out
        points_surf[self.num_samples_surf // 2:] += 0.03 * np.random.randn(self.num_samples_surf // 2, 3)
        sdfs_surf = np.zeros((self.num_samples_surf, 1))
        sdfs_surf[self.num_samples_surf // 2:] = -self.sdf_fn(points_surf[self.num_samples_surf // 2:])[:, None]
        
        # sdfs_surf = np.zeros((self.num_samples_surf, 1))
        
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

        # Call pysdf to compute the ground-truth SDF. 注意这里的符号约定取决于你的实现
        sdf_flat = -self.sdf_fn(pts_flat)  # shape: (res²,)
        sdf_slice = sdf_flat.reshape(resolution, resolution)

        # 确定剩下的两个维度对应的标签
        dims = ['x', 'y', 'z']
        dims.remove(axis)   # 比如 axis='z' 时，dims → ['x','y']
        xlabel, ylabel = dims  # 横轴、纵轴标签

        plt.figure(figsize=(6, 5))
        # ------- 这里不要再用 .T --------
        im = plt.imshow(
            sdf_slice,
            origin='lower',
            extent=[v0, v1, v0, v1],
            cmap=cmap,
            vmin=np.min(sdf_slice),
            vmax=np.max(sdf_slice)
        )
        plt.colorbar(im, label='SDF value')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"GT SDF slice on {axis}={coord:.2f}")
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_slice_dataset_{axis}.png", dpi=300)
        plt.close()

    def plot_dataset_sdf_binary_slice(self,
                                    workspace: str = None,
                                    axis: str = 'z',
                                    coord: float = 0.0,
                                    resolution: int = 256,
                                    val_range: tuple = (-1.0, 1.0),
                                    show_contour: bool = True):
        """
        绘制SDF的二值化切面图，显示SDF大于0和小于0的区域
        
        Parameters:
        workspace: 保存图片的目录
        axis: 切面轴向 ('x', 'y', 'z')
        coord: 切面坐标值
        resolution: 网格分辨率
        val_range: 采样范围 (min, max)
        show_contour: 是否显示SDF=0的等值线（物体表面）
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

        # 计算SDF值
        sdf_flat = -self.sdf_fn(pts_flat)  # shape: (res²,)
        sdf_slice = sdf_flat.reshape(resolution, resolution)

        # 创建二值化图像：SDF > 0 为外部区域，SDF < 0 为内部区域
        binary_slice = np.zeros_like(sdf_slice)
        binary_slice[sdf_slice > 0] = 1   # 外部区域
        binary_slice[sdf_slice < 0] = -1  # 内部区域
        
        # 确定剩下的两个维度对应的标签
        dims = ['x', 'y', 'z']
        dims.remove(axis)
        xlabel, ylabel = dims

        plt.figure(figsize=(8, 6))
        
        # 使用自定义颜色映射：蓝色表示内部(SDF<0)，红色表示外部(SDF>0)
        from matplotlib.colors import ListedColormap
        colors = ['blue', 'white', 'red']  # 内部、表面、外部
        cmap = ListedColormap(colors)
        
        im = plt.imshow(
            binary_slice,
            origin='lower',
            extent=[v0, v1, v0, v1],
            cmap=cmap,
            vmin=-1,
            vmax=1
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ticks=[-1, 0, 1])
        cbar.set_ticklabels(['Inside (SDF < 0)', 'Surface (SDF ≈ 0)', 'Outside (SDF > 0)'])
        
        # 如果需要，绘制SDF=0的等值线（物体表面）
        if show_contour:
            contour = plt.contour(
                np.linspace(v0, v1, resolution),
                np.linspace(v0, v1, resolution),
                sdf_slice,
                levels=[0],
                colors=['black'],
                linewidths=2
            )
            plt.clabel(contour, inline=True, fontsize=8, fmt='Surface')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"SDF Binary Regions Slice {axis}={coord:.2f}")
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_binary_slice_dataset_{axis}.png", dpi=300)
        plt.close()

    def plot_all_sdf_slices(self, workspace: str = None, coord: float = 0.0):
        """
        生成 x、y、z 三个方向的 SDF 切面图
        
        Parameters:
        workspace: 保存图片的目录
        coord: 切面的坐标值
        """
        for axis in ['x', 'y', 'z']:
            self.plot_dataset_sdf_slice(workspace=workspace, axis=axis, coord=coord)

    def plot_all_sdf_binary_slices(self, workspace: str = None, coord: float = 0.0, show_contour: bool = True):
        """
        生成 x、y、z 三个方向的 SDF 二值化切面图
        
        Parameters:
        workspace: 保存图片的目录
        coord: 切面的坐标值
        show_contour: 是否显示物体表面等值线
        """
        for axis in ['x', 'y', 'z']:
            self.plot_dataset_sdf_binary_slice(workspace=workspace, axis=axis, coord=coord, show_contour=show_contour)