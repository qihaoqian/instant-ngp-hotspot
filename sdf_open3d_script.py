import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import os


class Open3DSDFComputer:
    def __init__(self, path):
        """
        使用Open3D计算mesh的SDF值
        
        Args:
            path: mesh文件路径
        """
        self.path = path
        
        # 使用Open3D加载mesh
        print(f"[INFO] 加载mesh文件: {path}")
        self.o3d_mesh = o3d.io.read_triangle_mesh(path)
        
        if len(self.o3d_mesh.vertices) == 0:
            raise ValueError(f"无法加载mesh文件: {path}")
        
        print(f"[INFO] mesh: {len(self.o3d_mesh.vertices)} vertices, {len(self.o3d_mesh.triangles)} faces")
        
        # 检查mesh是否watertight（仅提示，不修复）
        if not self.o3d_mesh.is_watertight():
            print(f"[WARN] mesh不是watertight，但Open3D可以处理此情况")
        else:
            print(f"[INFO] mesh是watertight，SDF计算将更加准确")
        
        # 归一化mesh到[-1, 1]范围内
        self._normalize_mesh()
        
        # 创建SDF场景
        print("[INFO] 创建SDF场景...")
        self.mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(self.o3d_mesh)
        self.sdf_scene = o3d.t.geometry.RaycastingScene()
        self.sdf_scene.add_triangles(self.mesh_legacy)
        
        print("[INFO] SDF计算器初始化完成")

    def _normalize_mesh(self):
        """
        将mesh归一化到[-1, 1]范围内
        """
        vertices = np.asarray(self.o3d_mesh.vertices)
        
        # 计算边界框
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        center = (min_bound + max_bound) / 2
        
        # 计算缩放因子，使对象适合在单位球内
        bbox_diagonal = np.linalg.norm(max_bound - min_bound)
        scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
        
        # 应用变换
        vertices = (vertices - center) * scale_factor
        self.o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        print(f"[INFO] Mesh已归一化, 缩放因子: {scale_factor:.4f}")

    def compute_sdf(self, points):
        """
        计算给定点的SDF值
        
        Args:
            points: numpy数组, shape (N, 3)
            
        Returns:
            sdf_values: numpy数组, shape (N,)
        """
        # 转换为Open3D tensor格式
        query_points = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        
        # 计算有符号距离
        signed_distances = self.sdf_scene.compute_signed_distance(query_points, 8, 49)
        
        # 转换回numpy数组
        sdf_values = signed_distances.numpy()
        
        return sdf_values

    def plot_sdf_slice(self, 
                      workspace=None,
                      axis='z', 
                      coord=0.0, 
                      resolution=256, 
                      val_range=(-1.0, 1.0), 
                      cmap='jet'):
        """
        绘制SDF切面图
        """
        # 生成2D网格
        v0, v1 = val_range
        grid = np.linspace(v0, v1, resolution)
        A, B = np.meshgrid(grid, grid, indexing='xy')
        
        # 根据切面轴向扩展为3D点集
        if axis == 'x':
            pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
        elif axis == 'y':
            pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
        elif axis == 'z':
            pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
        else:
            raise ValueError(f"不支持的轴向 {axis}")
        
        pts_flat = pts.reshape(-1, 3).astype(np.float64)
        
        # 计算SDF值
        sdf_flat = self.compute_sdf(pts_flat)
        sdf_slice = sdf_flat.reshape(resolution, resolution)
        
        # 确定剩余两个维度的标签
        dims = ['x', 'y', 'z']
        dims.remove(axis)
        xlabel, ylabel = dims
        
        plt.figure(figsize=(6, 5))
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
        plt.title(f"Open3D GT SDF slice on {axis}={coord:.2f}")
        plt.tight_layout()
        
        if workspace:
            plt.savefig(f"{workspace}/open3d_sdf_slice_{axis}.png", dpi=300)
        plt.close()

    def plot_sdf_binary_slice(self,
                             workspace=None,
                             axis='z',
                             coord=0.0,
                             resolution=256,
                             val_range=(-1.0, 1.0),
                             show_contour=True):
        """
        绘制SDF二值化切面图
        """
        # 生成2D网格
        v0, v1 = val_range
        grid = np.linspace(v0, v1, resolution)
        A, B = np.meshgrid(grid, grid, indexing='xy')
        
        # 根据切面轴向扩展为3D点集
        if axis == 'x':
            pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
        elif axis == 'y':
            pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
        elif axis == 'z':
            pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
        else:
            raise ValueError(f"不支持的轴向 {axis}")
        
        pts_flat = pts.reshape(-1, 3).astype(np.float64)
        
        # 计算SDF值
        sdf_flat = self.compute_sdf(pts_flat)
        sdf_slice = sdf_flat.reshape(resolution, resolution)
        
        # 创建二值化图像
        binary_slice = np.zeros_like(sdf_slice)
        binary_slice[sdf_slice > 0] = 1   # 外部区域
        binary_slice[sdf_slice < 0] = -1  # 内部区域
        
        # 确定剩余两个维度的标签
        dims = ['x', 'y', 'z']
        dims.remove(axis)
        xlabel, ylabel = dims
        
        plt.figure(figsize=(8, 6))
        
        # 使用自定义颜色映射
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
        
        # 如果需要，绘制SDF=0的等值线
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
        plt.title(f"Open3D SDF Binary Regions Slice {axis}={coord:.2f}")
        plt.tight_layout()
        
        if workspace:
            plt.savefig(f"{workspace}/open3d_sdf_binary_slice_{axis}.png", dpi=300)
        plt.close()

    def plot_all_sdf_combined(self, workspace=None, coord=0.0, show_contour=True):
        """
        生成所有6张SDF图片（普通+二值化），拼接成2行3列的图片保存
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 使用自定义颜色映射用于二值化图
        colors = ['blue', 'white', 'red']  # 内部、表面、外部
        binary_cmap = ListedColormap(colors)
        
        for i, axis in enumerate(['x', 'y', 'z']):
            # 生成2D网格
            v0, v1 = (-1.0, 1.0)
            resolution = 256
            grid = np.linspace(v0, v1, resolution)
            A, B = np.meshgrid(grid, grid, indexing='xy')
            
            # 根据切面轴向扩展为3D点集
            if axis == 'x':
                pts = np.stack([np.full_like(A, coord), A, B], axis=-1)
            elif axis == 'y':
                pts = np.stack([A, np.full_like(A, coord), B], axis=-1)
            elif axis == 'z':
                pts = np.stack([A, B, np.full_like(A, coord)], axis=-1)
            
            pts_flat = pts.reshape(-1, 3).astype(np.float64)
            sdf_flat = self.compute_sdf(pts_flat)
            sdf_slice = sdf_flat.reshape(resolution, resolution)
            
            # 确定剩余两个维度的标签
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
            axes[0, i].set_title(f"Open3D GT SDF slice on {axis}={coord:.2f}")
            
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
            axes[1, i].set_title(f"Open3D SDF Binary Regions Slice {axis}={coord:.2f}")
        
        plt.tight_layout()
        if workspace:
            # 从mesh路径中提取文件名（不带扩展名）
            mesh_name = os.path.splitext(os.path.basename(self.path))[0]
            output_path = f"{workspace}/{mesh_name}_open3d_sdf_combined.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] 已保存组合SDF可视化图到: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='使用Open3D计算mesh的GT SDF并生成可视化')
    parser.add_argument('--mesh_path', type=str, required=True, 
                       help='mesh文件路径 (.obj, .ply, .stl等)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--coord', type=float, default=0.0,
                       help='切面坐标值')
    parser.add_argument('--resolution', type=int, default=256,
                       help='网格分辨率')
    parser.add_argument('--show_contour', action='store_true',
                       help='是否显示物体表面等值线')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 初始化SDF计算器
        print(f"[INFO] 初始化Open3D SDF计算器...")
        sdf_computer = Open3DSDFComputer(args.mesh_path)
        
        # 生成组合可视化图
        print(f"[INFO] 生成SDF可视化图...")
        sdf_computer.plot_all_sdf_combined(
            workspace=args.output_dir,
            coord=args.coord,
            show_contour=args.show_contour
        )
        
        print(f"[INFO] ✓ SDF可视化图已生成完成!")
        print(f"[INFO] 输出目录: {args.output_dir}")
        
    except Exception as e:
        print(f"[ERROR] 执行失败: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 