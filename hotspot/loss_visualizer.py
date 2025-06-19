#!/usr/bin/env python3
"""
SDF网络损失可视化脚本
基于compute_projection_loss.py的结构，提供完整的损失可视化功能
现在使用切片可视化方式，类似compute_projection_loss.py
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import open3d as o3d
import os, sys
import torch

# Add project_root to import search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import Config
from hotspot.network import SDFNetwork
from hotspot.utils import *


def load_model(checkpoint_path, config_path):
    """Load trained model"""
    parser = Config.get_argparser()
    cfg, _ = parser.parse_known_args(["--config", config_path])
    seed_everything(cfg.seed)
    
    if cfg.model.encoding == "hashgrid":
        encoding_config = cfg.hash_grid
    elif cfg.model.encoding == "reg_grid":
        encoding_config = cfg.reg_grid
    else:
        encoding_config = None

    model = SDFNetwork(encoding=cfg.model.encoding, 
                    encoding_config=encoding_config,
                    num_layers=cfg.model.num_layers, 
                    hidden_dim=cfg.model.hidden_dim,
                    sphere_radius=cfg.model.sphere_radius,
                    sphere_scale=cfg.model.sphere_scale,
                    use_sphere_post_processing=cfg.model.use_sphere_post_processing,
                    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return model.eval(), device


def finite_diff_grad(model, X, h1=1e-4):
    """Compute finite difference gradients"""
    grads = []
    valid_min = torch.tensor([-1,-1,-1], device=X.device) + h1
    valid_max = torch.tensor([1,1,1], device=X.device) - h1
    mask = ((X > valid_min) & (X < valid_max)).all(dim=1)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    X_safe = X[idx]
    
    for i in range(X_safe.shape[1]):
        offset = torch.zeros_like(X_safe)
        offset[:, i] = h1
        sdf_plus = model(X_safe + offset)
        sdf_minus = model(X_safe - offset)
        grad_i = (sdf_plus - sdf_minus) / (2 * h1)
        grads.append(grad_i)
    
    grad = torch.cat(grads, dim=-1)
    return grad, idx


def generate_slice_points(slice_axis=0, slice_position=0.0, resolution=256):
    """
    生成切片上的点
    Args:
        slice_axis: 0=YZ plane (fixed X), 1=XZ plane (fixed Y), 2=XY plane (fixed Z)
        slice_position: Position of the slice on the corresponding axis [-1,1]
        resolution: Slice resolution
    """
    coords = np.linspace(-1, 1, resolution, dtype=np.float32)
    
    if slice_axis == 0:  # YZ plane (fixed X)
        Y, Z = np.meshgrid(coords, coords, indexing='ij')
        X = np.full_like(Y, slice_position)
        slice_name = f"YZ_slice_X_{slice_position:.2f}"
    elif slice_axis == 1:  # XZ plane (fixed Y)
        X, Z = np.meshgrid(coords, coords, indexing='ij')
        Y = np.full_like(X, slice_position)
        slice_name = f"XZ_slice_Y_{slice_position:.2f}"
    else:  # XY plane (fixed Z)
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        Z = np.full_like(X, slice_position)
        slice_name = f"XY_slice_Z_{slice_position:.2f}"
    
    # Convert slice points to tensor
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    X_slice = torch.from_numpy(pts).float()
    
    return X_slice, slice_name, resolution


def compute_boundary_loss_slice(model, mesh_path, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', num_samples_surf=5000):
    """计算切片上的边界损失 - 采用从mesh sample的方法获取boundary points"""
    # Load and preprocess mesh (和dataset保持一致的预处理)
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    
    # Generate slice points (用于可视化背景)
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Surface sampling - 参考dataset的实现
    points_surf = mesh.sample(num_samples_surf)  # 从mesh表面采样点
    # 对一部分点添加小的扰动 (参考dataset)
    points_surf[num_samples_surf // 2:] += 0.03 * np.random.randn(num_samples_surf // 2, 3)
    
    # 筛选出在切片附近的surface points (在指定axis上接近slice_position的点)
    tolerance = 0.1  # 切片厚度的一半
    if slice_axis == 0:  # YZ plane (fixed X)
        surface_mask = np.abs(points_surf[:, 0] - slice_position) < tolerance
    elif slice_axis == 1:  # XZ plane (fixed Y)  
        surface_mask = np.abs(points_surf[:, 1] - slice_position) < tolerance
    else:  # XY plane (fixed Z)
        surface_mask = np.abs(points_surf[:, 2] - slice_position) < tolerance
    
    points_surf_slice = points_surf[surface_mask]
    
    if len(points_surf_slice) == 0:
        print(f"[WARNING] No surface points found near slice {slice_name}")
        boundary_loss_grid = np.full((res, res), np.nan)
        return boundary_loss_grid, slice_name
    
    # 将surface points转换为tensor
    points_surf_tensor = torch.from_numpy(points_surf_slice).float().to(device)
    
    # Predict SDF values for surface points (这些点应该接近0)
    with torch.no_grad():
        sdf_pred_surf = model(points_surf_tensor)
    
    # Boundary loss: surface points的SDF应该接近0
    boundary_loss_values = torch.abs(sdf_pred_surf.squeeze())
    
    # 将surface points投影到2D切片坐标系统上进行可视化
    if slice_axis == 0:  # YZ plane
        surf_2d = points_surf_slice[:, [1, 2]]  # Y, Z coordinates
    elif slice_axis == 1:  # XZ plane
        surf_2d = points_surf_slice[:, [0, 2]]  # X, Z coordinates  
    else:  # XY plane
        surf_2d = points_surf_slice[:, [0, 1]]  # X, Y coordinates
    
    # 创建可视化网格
    boundary_loss_grid = np.full((res, res), np.nan)
    
    # 将surface points的loss映射到网格上
    for i, (point_2d, loss_val) in enumerate(zip(surf_2d, boundary_loss_values.cpu().numpy())):
        # 将[-1,1]范围的坐标映射到[0,res-1]的网格索引
        grid_x = int((point_2d[0] + 1) / 2 * (res - 1))
        grid_y = int((point_2d[1] + 1) / 2 * (res - 1))
        
        # 确保索引在有效范围内
        if 0 <= grid_x < res and 0 <= grid_y < res:
            # 如果该网格点已有值，取平均值
            if not np.isnan(boundary_loss_grid[grid_y, grid_x]):
                boundary_loss_grid[grid_y, grid_x] = (boundary_loss_grid[grid_y, grid_x] + loss_val) / 2
            else:
                boundary_loss_grid[grid_y, grid_x] = loss_val
    
    print(f"[INFO] {slice_name}: Found {len(points_surf_slice)} surface points, "
          f"mean boundary loss: {boundary_loss_values.mean().item():.6f}")
    
    return boundary_loss_grid, slice_name


def compute_eikonal_loss_slice(model, slice_axis=0, slice_position=0.0, resolution=256, device='cuda'):
    """计算切片上的Eikonal损失"""
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Compute gradients
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-2)
    
    # Compute gradient norm
    grad_norm = torch.norm(grad_slice, dim=1)
    
    # Compute Eikonal loss (should be close to 1)
    eikonal_loss = torch.abs(grad_norm - 1)
    # eikonal_loss = grad_norm
    
    # Create full grid and fill with valid values
    eikonal_loss_grid = torch.full((X_slice.shape[0],), float('nan'), device=device)
    eikonal_loss_grid[grad_idx] = eikonal_loss
    eikonal_loss_grid = eikonal_loss_grid.view(res, res)
    
    return eikonal_loss_grid.detach().cpu().numpy(), slice_name


def compute_gradient_slice(model, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', subsample=4):
    """计算切片上的梯度，返回方向和大小信息用于箭头可视化"""
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Compute gradients
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-2)
    
    # Create full grid for gradients
    grad_grid = torch.full((X_slice.shape[0], 3), float('nan'), device=device)
    grad_grid[grad_idx] = grad_slice
    grad_grid = grad_grid.view(res, res, 3)
    
    # Convert to numpy
    grad_grid_np = grad_grid.detach().cpu().numpy()
    
    # 根据切片轴选择需要显示的梯度分量
    if slice_axis == 0:  # YZ plane (fixed X), 显示Y和Z方向的梯度
        grad_u = grad_grid_np[:, :, 1]  # Y component  
        grad_v = grad_grid_np[:, :, 2]  # Z component
        axis_names = ('Y', 'Z')
    elif slice_axis == 1:  # XZ plane (fixed Y), 显示X和Z方向的梯度
        grad_u = grad_grid_np[:, :, 0]  # X component
        grad_v = grad_grid_np[:, :, 2]  # Z component
        axis_names = ('X', 'Z')
    else:  # XY plane (fixed Z), 显示X和Y方向的梯度
        grad_u = grad_grid_np[:, :, 0]  # X component
        grad_v = grad_grid_np[:, :, 1]  # Y component
        axis_names = ('X', 'Y')
    
    # 计算梯度大小
    grad_magnitude = np.sqrt(grad_u**2 + grad_v**2)
    
    # 为箭头图准备坐标网格，进行子采样以避免过于密集
    coords = np.linspace(-1, 1, res)
    Y_grid, X_grid = np.meshgrid(coords, coords, indexing='ij')
    
    # 子采样以减少箭头数量
    step = subsample
    Y_sub = Y_grid[::step, ::step]
    X_sub = X_grid[::step, ::step]
    grad_u_sub = grad_u[::step, ::step]
    grad_v_sub = grad_v[::step, ::step]
    grad_mag_sub = grad_magnitude[::step, ::step]
    
    return {
        'grad_u': grad_u,
        'grad_v': grad_v, 
        'grad_magnitude': grad_magnitude,
        'Y_sub': Y_sub,
        'X_sub': X_sub,
        'grad_u_sub': grad_u_sub,
        'grad_v_sub': grad_v_sub,
        'grad_mag_sub': grad_mag_sub,
        'slice_name': slice_name,
        'axis_names': axis_names
    }


def compute_sign_loss_slice(model, mesh_path, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', surface_threshold=0.05):
    """计算切片上的符号损失 - 只在自由空间（free points）上计算"""
    # Load and preprocess mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Compute ground truth SDF values for slice points using Open3D
    # 设置Open3D的RaycastingScene
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    
    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    raycasting_scene.add_triangles(o3d_mesh)
    
    # 计算SDF值
    query_points = o3d.core.Tensor(X_slice.cpu().numpy().astype(np.float32), dtype=o3d.core.Dtype.Float32)
    sdf_values = raycasting_scene.compute_signed_distance(query_points)
    sdf_gt = torch.from_numpy(-sdf_values.numpy()).float().to(device)  # 注意符号，取反保持一致
    
    # 只在自由空间计算符号损失：SDF > threshold
    free_mask = sdf_gt > surface_threshold  # 自由空间：SDF > threshold
    
    # Predict SDF values
    with torch.no_grad():
        sdf_pred = model(X_slice)
    
    # 初始化loss grid
    sign_loss_grid = torch.full((res * res,), float('nan'), device=device)
    
    # 只在自由空间计算符号损失
    if free_mask.sum() > 0:
        free_indices = torch.where(free_mask)[0]
        # 对于自由空间的点，如果预测为负值则是错误
        free_pred = sdf_pred.squeeze()[free_indices]
        # 使用ReLU来计算损失强度：预测越负，损失越大
        sign_loss_values = torch.clamp(-free_pred, min=0)  # 只有当预测<0时才有损失
        sign_loss_grid[free_indices] = sign_loss_values
    
    sign_loss_grid = sign_loss_grid.view(res, res)
    
    return sign_loss_grid.detach().cpu().numpy(), slice_name


def compute_heat_loss_slice(model, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', heat_lambda=4):
    """计算切片上的热损失"""
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Predict SDF values
    with torch.no_grad():
        sdf_pred = model(X_slice)
    
    # Compute gradients
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-4)
    
    # Compute heat loss terms
    heat = torch.exp(-heat_lambda * torch.abs(sdf_pred.squeeze()))
    grad_norm_sq = torch.norm(grad_slice, dim=1) ** 2
    
    # Heat loss formula: 0.5 * heat^2 * (grad_norm^2 + 1)
    heat_loss_values = 0.5 * heat[grad_idx] ** 2 * (grad_norm_sq + 1)
    
    # Create full grid and fill with valid values
    heat_loss_grid = torch.full((X_slice.shape[0],), float('nan'), device=device)
    heat_loss_grid[grad_idx] = heat_loss_values
    heat_loss_grid = heat_loss_grid.view(res, res)
    
    return heat_loss_grid.detach().cpu().numpy(), slice_name


def compute_projection_loss_slice(model, mesh_path, slice_axis=0, slice_position=0.0, resolution=256, device='cuda'):
    """计算切片上的投影损失"""
    # Load and preprocess mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    
    # Generate surface points for nearest neighbor computation
    surface_points = mesh.sample(count=50000)
    X_surf = torch.from_numpy(surface_points).float().to(device)
    
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Predict SDF values
    with torch.no_grad():
        sdf_pred = model(X_slice)
    
    # Compute gradients
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-2)
    X_slice_safe = X_slice[grad_idx]
    
    # 初始化loss grid
    projection_loss_grid = torch.full((X_slice.shape[0],), float('nan'), device=device)
    
    if len(grad_idx) > 0:
        # 计算每个空间点到所有surface点的距离，找到最近邻
        d2_space = torch.cdist(X_slice_safe, X_surf)  # (N_space, N_surf)
        nn_idx_space = d2_space.argmin(dim=1)
        
        # 构造ground-truth方向向量
        X_nn_space = X_surf[nn_idx_space]  # (N_space, 3)
        dir_gt_space = X_slice_safe - X_nn_space  # (N_space, 3)
        dir_gt_space_norm = dir_gt_space / (dir_gt_space.norm(dim=1, keepdim=True) + 1e-8)
        
        # 计算投影点
        space_pred = sdf_pred[grad_idx]
        X_proj = X_slice_safe - dir_gt_space_norm * space_pred.abs()
        
        # 计算投影点到surface的最小距离
        dists = torch.cdist(X_proj, X_surf).min(dim=1)[0]
        projection_loss_grid[grad_idx] = dists.abs()
    
    projection_loss_grid = projection_loss_grid.view(res, res)
    
    return projection_loss_grid.detach().cpu().numpy(), slice_name


def compute_sec_grad_loss_slice(model, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', h1=1e-2, h2=1e-2):
    """计算切片上的二阶梯度损失"""
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # 计算二阶导数损失的安全区域
    valid_min = torch.tensor([-1,-1,-1], device=X_slice.device) + h1 + h2
    valid_max = torch.tensor([1,1,1], device=X_slice.device) - h1 - h2
    safe_mask = ((X_slice > valid_min) & (X_slice < valid_max)).all(dim=1)
    safe_idx = torch.where(safe_mask)[0]
    X_safe = X_slice[safe_idx]
    
    # 初始化loss grid
    sec_grad_loss_grid = torch.full((X_slice.shape[0],), float('nan'), device=device)
    
    if len(safe_idx) > 0:
        # 计算二阶导数
        grads_1 = []
        grads_2 = []
        
        for i in range(3):  # x, y, z three dimensions
            offset1 = torch.zeros_like(X_safe)
            offset2 = torch.zeros_like(X_safe)
            offset1[:, i] = h1
            offset2[:, i] = h2
            
            # 计算一阶导数在不同位置的值
            grad_1 = (model(X_safe + offset2 + offset1) - model(X_safe + offset2 - offset1)) / (2 * h1)
            grad_2 = (model(X_safe - offset2 + offset1) - model(X_safe - offset2 - offset1)) / (2 * h1)
            grads_1.append(grad_1)
            grads_2.append(grad_2)
        
        grads_1 = torch.cat(grads_1, dim=-1)  # [B, 3]
        grads_2 = torch.cat(grads_2, dim=-1)  # [B, 3]
        
        # 计算二阶导数
        sec_grads = (grads_1 - grads_2) / (2 * h2)  # [B, 3]
        
        # 计算二阶梯度损失：只对绝对值大于阈值的部分计算损失
        sec_grad_loss_values = ((sec_grads.abs() >= 0.5).float() * sec_grads).norm(dim=-1)
        sec_grad_loss_grid[safe_idx] = sec_grad_loss_values
    
    sec_grad_loss_grid = sec_grad_loss_grid.view(res, res)
    
    return sec_grad_loss_grid.detach().cpu().numpy(), slice_name


def compute_grad_dir_loss_slice(model, mesh_path, slice_axis=0, slice_position=0.0, resolution=256, device='cuda', num_samples_surf=10000):
    """计算切片上的梯度方向损失 - 计算空间点梯度方向与最近surface点梯度方向的一致性"""
    # Load and preprocess mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    
    # Generate slice points
    X_slice, slice_name, res = generate_slice_points(slice_axis, slice_position, resolution)
    X_slice = X_slice.to(device)
    
    # Surface sampling for nearest neighbor computation
    surface_points = mesh.sample(num_samples_surf)
    X_surf = torch.from_numpy(surface_points).float().to(device)
    
    # Compute gradients for slice points
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-2)
    X_slice_safe = X_slice[grad_idx]
    
    # Compute gradients for surface points
    grad_surf, grad_surf_idx = finite_diff_grad(model, X_surf, h1=1e-2)
    X_surf_safe = X_surf[grad_surf_idx]
    
    # 初始化loss grid
    grad_dir_loss_grid = torch.full((X_slice.shape[0],), float('nan'), device=device)
    
    if len(grad_idx) > 0 and len(grad_surf_idx) > 0:
        # 计算每个空间点到所有有效surface点的距离，找到最近邻
        d2_space = torch.cdist(X_slice_safe, X_surf_safe)  # (N_space, N_surf_safe)
        nn_idx_space = d2_space.argmin(dim=1)
        
        # 获取最近邻surface点的梯度方向作为ground-truth
        grad_nn_space = grad_surf[nn_idx_space]  # (N_space, 3)
        
        # 计算梯度方向的归一化
        grad_space_norm = grad_slice / (grad_slice.norm(dim=1, keepdim=True) + 1e-8)
        grad_nn_space_norm = grad_nn_space / (grad_nn_space.norm(dim=1, keepdim=True) + 1e-8)
        
        # 计算点积，值越接近1表示方向越一致
        dot_product = (grad_space_norm * grad_nn_space_norm).sum(dim=1)
        
        # 梯度方向损失：1 - dot_product，值越小表示方向越一致
        grad_dir_loss_values = 1.0 - dot_product
        
        grad_dir_loss_grid[grad_idx] = grad_dir_loss_values
    
    grad_dir_loss_grid = grad_dir_loss_grid.view(res, res)
    
    return grad_dir_loss_grid.detach().cpu().numpy(), slice_name


def visualize_all_losses_slices(checkpoint_path, config_path, mesh_path, save_dir="loss_visualizations", 
                               resolution=128, epoch=None):
    """可视化所有损失的切片"""
    print("加载模型...")
    model, device = load_model(checkpoint_path, config_path)
    
    # 如果没有指定epoch，尝试从checkpoint文件名或内容中读取
    if epoch is None:
        import re
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'ep(\d+)', filename)
        if match:
            epoch = int(match.group(1))
            print(f"从checkpoint文件名中解析到epoch: {epoch}")
        else:
            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                if 'epoch' in ckpt:
                    epoch = ckpt['epoch']
                    print(f"从checkpoint内容中读取到epoch: {epoch}")
                else:
                    epoch = 0
                    print("checkpoint中未找到epoch信息，使用默认值0")
            except:
                epoch = 0
                print("无法读取checkpoint信息，使用默认值0")
    
    # 定义切片信息
    slices_info = [
        (0, 0.0, "YZ Plane (X=0.0)"),  # YZ plane
        (1, 0.0, "XZ Plane (Y=0.0)"),  # XZ plane  
        (2, 0.0, "XY Plane (Z=0.0)")   # XY plane
    ]
    
    # 构建文件名后缀
    suffix = f"_ep{epoch:04d}" if epoch is not None else ""
    
    print("计算各种损失的切片可视化...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 边界损失
    print("计算边界损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        boundary_loss, slice_name = compute_boundary_loss_slice(
            model, mesh_path, axis, pos, resolution, device)
        
        im = axes[i].imshow(boundary_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='hot', interpolation='nearest')
        axes[i].set_title(f'{title}\nBoundary Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"boundary_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Eikonal损失
    print("计算Eikonal损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        eikonal_loss, slice_name = compute_eikonal_loss_slice(
            model, axis, pos, resolution, device)
        
        im = axes[i].imshow(eikonal_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'{title}\nEikonal Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"eikonal_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 符号损失（只在自由空间计算）
    print("计算符号损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        sign_loss, slice_name = compute_sign_loss_slice(
            model, mesh_path, axis, pos, resolution, device)
        
        # 使用热力图显示损失强度，错误越大颜色越红
        im = axes[i].imshow(sign_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='Reds', interpolation='nearest')
        axes[i].set_title(f'{title}\nSign Loss (Free Space Only)')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label('Sign Loss Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sign_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 热损失
    print("计算热损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        heat_loss, slice_name = compute_heat_loss_slice(
            model, axis, pos, resolution, device)
        
        im = axes[i].imshow(heat_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='plasma', interpolation='nearest')
        axes[i].set_title(f'{title}\nHeat Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heat_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 投影损失
    print("计算投影损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        projection_loss, slice_name = compute_projection_loss_slice(
            model, mesh_path, axis, pos, resolution, device)
        
        im = axes[i].imshow(projection_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='hot', interpolation='nearest')
        axes[i].set_title(f'{title}\nProjection Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"projection_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 二阶梯度损失
    print("计算二阶梯度损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        sec_grad_loss, slice_name = compute_sec_grad_loss_slice(
            model, axis, pos, resolution, device)
        
        im = axes[i].imshow(sec_grad_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='inferno', interpolation='nearest')
        axes[i].set_title(f'{title}\nSecond Gradient Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sec_grad_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 梯度向量场可视化
    print("计算梯度切片可视化...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 增大画布尺寸
    for i, (axis, pos, title) in enumerate(slices_info):
        # 使用更高分辨率计算梯度场
        grad_data = compute_gradient_slice(model, axis, pos, resolution*2, device, subsample=6)
        
        # 背景显示梯度大小
        im = axes[i].imshow(grad_data['grad_magnitude'].T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='viridis', alpha=0.7, interpolation='bilinear')
        
        # 绘制梯度方向箭头
        # 过滤掉无效值
        valid_mask = ~(np.isnan(grad_data['grad_u_sub']) | np.isnan(grad_data['grad_v_sub']))
        if np.any(valid_mask):
            # 标准化箭头长度，使其更好可视化
            scale_factor = 0.15  # 调整箭头长度，更小
            quiver = axes[i].quiver(
                grad_data['X_sub'][valid_mask], 
                grad_data['Y_sub'][valid_mask],
                grad_data['grad_u_sub'][valid_mask] * scale_factor,
                grad_data['grad_v_sub'][valid_mask] * scale_factor,
                grad_data['grad_mag_sub'][valid_mask],
                cmap='plasma', 
                alpha=0.8,
                scale_units='xy', 
                scale=1,
                width=0.002
            )
        
        axes[i].set_title(f'{title}\nGradient Field')
        axes[i].set_xlabel(grad_data['axis_names'][0])
        axes[i].set_ylabel(grad_data['axis_names'][1])
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(-1, 1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label('Gradient Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gradient_field_slices{suffix}.png"), dpi=300, bbox_inches='tight')  # 提高DPI到600
    plt.close()
    
    # 8. 梯度方向损失
    print("计算梯度方向损失切片...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, pos, title) in enumerate(slices_info):
        grad_dir_loss, slice_name = compute_grad_dir_loss_slice(
            model, mesh_path, axis, pos, resolution, device)
        
        im = axes[i].imshow(grad_dir_loss.T, origin='lower', extent=[-1, 1, -1, 1], 
                           cmap='inferno', interpolation='nearest')
        axes[i].set_title(f'{title}\nGradient Direction Loss')
        
        if axis == 0:  # YZ plane
            axes[i].set_xlabel('Y')
            axes[i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Z')
        else:  # XY plane
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"grad_dir_loss_slices{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有损失切片可视化完成，保存在目录: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化SDF网络的各种损失切片')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--mesh', type=str, default='data/armadillo.obj', help='mesh文件路径')
    parser.add_argument('--save_dir', type=str, default='loss_visualizations', help='保存目录')
    parser.add_argument('--resolution', type=int, default=128, help='切片分辨率')
    parser.add_argument('--epoch', type=int, default=None, help='用于文件命名的epoch（可选，默认从checkpoint中读取）')
    
    args = parser.parse_args()
    
    print(f"模型checkpoint: {args.checkpoint}")
    print(f"配置文件: {args.config}")
    print(f"Mesh文件: {args.mesh}")
    print(f"保存目录: {args.save_dir}")
    print(f"分辨率: {args.resolution}")
    print(f"Epoch: {args.epoch}")
    
    visualize_all_losses_slices(args.checkpoint, args.config, args.mesh, 
                               args.save_dir, args.resolution, args.epoch)


if __name__ == "__main__":
    main() 