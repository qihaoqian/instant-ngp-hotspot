import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import open3d as o3d
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

        # 使用Open3D替代pysdf计算SDF
        self._setup_open3d_sdf()
        
        self.num_samples_surf = num_samples_surf
        self.num_samples_space = num_samples_space

        self.size = size

    def _setup_open3d_sdf(self):
        """
        设置Open3D的RaycastingScene来计算SDF
        """
        # 将trimesh转换为Open3D的TriangleMesh
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.astype(np.int32)
        
        # 创建Open3D tensor格式的mesh
        self.o3d_mesh = o3d.t.geometry.TriangleMesh()
        self.o3d_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
        self.o3d_mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
        
        # 创建RaycastingScene
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        self.raycasting_scene.add_triangles(self.o3d_mesh)
        
        print("[INFO] ✓ Open3D RaycastingScene已设置完成")
    
    def _compute_sdf_open3d(self, points):
        """
        使用Open3D计算SDF值
        
        Args:
            points: numpy array of shape (N, 3)
            
        Returns:
            numpy array of shape (N,) with SDF values
        """
        # 转换为Open3D tensor
        query_points = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        
        # 计算signed distance
        sdf_values = self.raycasting_scene.compute_signed_distance(query_points)
        
        # 返回numpy array，注意Open3D的SDF符号约定：正值表示外部，负值表示内部
        # 为了和pysdf保持一致（负值表示外部，正值表示内部），需要取反
        return -sdf_values.numpy()

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # Surface sampling
        points_surf = self.mesh.sample(self.num_samples_surf)
        
        # Perturbation of the surface is commented out
        points_surf[self.num_samples_surf // 2:] += 0.03 * np.random.randn(self.num_samples_surf // 2, 3)
        sdfs_surf = np.zeros((self.num_samples_surf, 1))
        sdfs_surf[self.num_samples_surf // 2:] = self._compute_sdf_open3d(points_surf[self.num_samples_surf // 2:])[:, None]
        
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
        
        sdfs_space = self._compute_sdf_open3d(points_space)[:, None]  # shape: (N, 1)

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

        # 使用Open3D计算SDF
        sdf_flat = self._compute_sdf_open3d(pts_flat)  # shape: (res²,)
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

        # 使用Open3D计算SDF
        sdf_flat = self._compute_sdf_open3d(pts_flat)  # shape: (res²,)
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
        生成 x、y、z 三个方向的 SDF 切面图，拼接成一张图片保存
        
        Parameters:
        workspace: 保存图片的目录
        coord: 切面的坐标值
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, axis in enumerate(['x', 'y', 'z']):
            # Generate a 2D grid
            v0, v1 = (-1.0, 1.0)
            resolution = 256
            grid = np.linspace(v0, v1, resolution)
            A, B = np.meshgrid(grid, grid, indexing='xy')
            
            # Expand the grid to a 3D point set based on the slicing axis
            if axis == 'x':
                pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
            elif axis == 'y':
                pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
            elif axis == 'z':
                pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
            
            pts_flat = pts.reshape(-1, 3).astype(np.float64)
            sdf_flat = self._compute_sdf_open3d(pts_flat)
            sdf_slice = sdf_flat.reshape(resolution, resolution)
            
            # 确定剩下的两个维度对应的标签
            dims = ['x', 'y', 'z']
            dims.remove(axis)
            xlabel, ylabel = dims
            
            im = axes[i].imshow(
                sdf_slice,
                origin='lower',
                extent=[v0, v1, v0, v1],
                cmap='jet',
                vmin=np.min(sdf_slice),
                vmax=np.max(sdf_slice)
            )
            plt.colorbar(im, ax=axes[i], label='SDF value')
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f"GT SDF slice on {axis}={coord:.2f}")
        
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_slices_combined.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_sdf_binary_slices(self, workspace: str = None, coord: float = 0.0, show_contour: bool = True):
        """
        生成 x、y、z 三个方向的 SDF 二值化切面图，拼接成一张图片保存
        
        Parameters:
        workspace: 保存图片的目录
        coord: 切面的坐标值
        show_contour: 是否显示物体表面等值线
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 使用自定义颜色映射
        from matplotlib.colors import ListedColormap
        colors = ['blue', 'white', 'red']  # 内部、表面、外部
        cmap = ListedColormap(colors)
        
        for i, axis in enumerate(['x', 'y', 'z']):
            # Generate a 2D grid
            v0, v1 = (-1.0, 1.0)
            resolution = 256
            grid = np.linspace(v0, v1, resolution)
            A, B = np.meshgrid(grid, grid, indexing='xy')
            
            # Expand the grid to a 3D point set based on the slicing axis
            if axis == 'x':
                pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
            elif axis == 'y':
                pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
            elif axis == 'z':
                pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
            
            pts_flat = pts.reshape(-1, 3).astype(np.float64)
            sdf_flat = self._compute_sdf_open3d(pts_flat)
            sdf_slice = sdf_flat.reshape(resolution, resolution)
            
            # 创建二值化图像
            binary_slice = np.zeros_like(sdf_slice)
            binary_slice[sdf_slice > 0] = 1   # 外部区域
            binary_slice[sdf_slice < 0] = -1  # 内部区域
            
            # 确定剩下的两个维度对应的标签
            dims = ['x', 'y', 'z']
            dims.remove(axis)
            xlabel, ylabel = dims
            
            im = axes[i].imshow(
                binary_slice,
                origin='lower',
                extent=[v0, v1, v0, v1],
                cmap=cmap,
                vmin=-1,
                vmax=1
            )
            
            # 只在最后一个子图添加颜色条
            if i == 2:
                cbar = plt.colorbar(im, ax=axes[i], ticks=[-1, 0, 1])
                cbar.set_ticklabels(['Inside (SDF < 0)', 'Surface (SDF ≈ 0)', 'Outside (SDF > 0)'])
            
            # 如果需要，绘制SDF=0的等值线
            if show_contour:
                contour = axes[i].contour(
                    np.linspace(v0, v1, resolution),
                    np.linspace(v0, v1, resolution),
                    sdf_slice,
                    levels=[0],
                    colors=['black'],
                    linewidths=2
                )
                axes[i].clabel(contour, inline=True, fontsize=8, fmt='Surface')
            
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f"SDF Binary Regions Slice {axis}={coord:.2f}")
        
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_binary_slices_combined.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_sdf_combined(self, workspace: str = None, coord: float = 0.0, show_contour: bool = True):
        """
        生成所有6张SDF图片（普通+二值化），拼接成2行3列的图片保存
        
        Parameters:
        workspace: 保存图片的目录
        coord: 切面的坐标值
        show_contour: 是否在二值化图中显示物体表面等值线
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 使用自定义颜色映射用于二值化图
        from matplotlib.colors import ListedColormap
        colors = ['blue', 'white', 'red']  # 内部、表面、外部
        binary_cmap = ListedColormap(colors)
        
        for i, axis in enumerate(['x', 'y', 'z']):
            # Generate a 2D grid
            v0, v1 = (-1.0, 1.0)
            resolution = 256
            grid = np.linspace(v0, v1, resolution)
            A, B = np.meshgrid(grid, grid, indexing='xy')
            
            # Expand the grid to a 3D point set based on the slicing axis
            if axis == 'x':
                pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
            elif axis == 'y':
                pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
            elif axis == 'z':
                pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
            
            pts_flat = pts.reshape(-1, 3).astype(np.float64)
            sdf_flat = self._compute_sdf_open3d(pts_flat)
            sdf_slice = sdf_flat.reshape(resolution, resolution)
            
            # 确定剩下的两个维度对应的标签
            dims = ['x', 'y', 'z']
            dims.remove(axis)
            xlabel, ylabel = dims
            
            # 第一行：普通SDF切面图
            im1 = axes[0, i].imshow(
                sdf_slice,
                origin='lower',
                extent=[v0, v1, v0, v1],
                cmap='jet',
                vmin=np.min(sdf_slice),
                vmax=np.max(sdf_slice)
            )
            plt.colorbar(im1, ax=axes[0, i], label='SDF value')
            axes[0, i].set_xlabel(xlabel)
            axes[0, i].set_ylabel(ylabel)
            axes[0, i].set_title(f"GT SDF slice on {axis}={coord:.2f}")
            
            # 第二行：二值化SDF切面图
            binary_slice = np.zeros_like(sdf_slice)
            binary_slice[sdf_slice > 0] = 1   # 外部区域
            binary_slice[sdf_slice < 0] = -1  # 内部区域
            
            im2 = axes[1, i].imshow(
                binary_slice,
                origin='lower',
                extent=[v0, v1, v0, v1],
                cmap=binary_cmap,
                vmin=-1,
                vmax=1
            )
            
            # 只在最后一个子图添加颜色条
            if i == 2:
                cbar = plt.colorbar(im2, ax=axes[1, i], ticks=[-1, 0, 1])
                cbar.set_ticklabels(['Inside (SDF < 0)', 'Surface (SDF ≈ 0)', 'Outside (SDF > 0)'])
            
            # 如果需要，绘制SDF=0的等值线
            if show_contour:
                contour = axes[1, i].contour(
                    np.linspace(v0, v1, resolution),
                    np.linspace(v0, v1, resolution),
                    sdf_slice,
                    levels=[0],
                    colors=['black'],
                    linewidths=2
                )
                axes[1, i].clabel(contour, inline=True, fontsize=8, fmt='Surface')
            
            axes[1, i].set_xlabel(xlabel)
            axes[1, i].set_ylabel(ylabel)
            axes[1, i].set_title(f"SDF Binary Regions Slice {axis}={coord:.2f}")
        
        plt.tight_layout()
        plt.savefig(f"{workspace}/sdf_all_slices_combined.png", dpi=300, bbox_inches='tight')
        plt.close()