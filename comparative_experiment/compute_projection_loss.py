import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
# import pysdf  # 不再使用pysdf，改为Open3D
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


def compute_projection_loss_slice(checkpoint_path, config_path, mesh_path='data/armadillo.obj', 
                                 slice_axis=0, slice_position=0.0, resolution=256):
    """
    Compute projection loss for a specific slice
    
    Args:
        slice_axis: 0=YZ plane (fixed X), 1=XZ plane (fixed Y), 2=XY plane (fixed Z)
        slice_position: Position of the slice on the corresponding axis [-1,1]
        resolution: Slice resolution
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)
    mesh.vertices -= center
    mesh.vertices *= scale_factor
    
    # Get surface points (sample from mesh surface)
    surface_points = mesh.sample(count=50000)
    X_surf = torch.from_numpy(surface_points).float()
    
    # Generate points on the slice
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
    
    # Load model and predict SDF
    model, device = load_model(checkpoint_path, config_path)
    X_surf = X_surf.to(device)
    X_slice = X_slice.to(device)
    
    # Predict SDF values
    with torch.no_grad():
        sdf_pred = model(X_slice).view(-1)
    
    # Compute gradients
    grad_slice, grad_idx = finite_diff_grad(model, X_slice, h1=1e-4)
    X_slice_safe = X_slice[grad_idx]
    
    # Compute projection loss
    with torch.no_grad():
        # Find nearest surface point for each point
        d2_slice = torch.cdist(X_slice_safe, X_surf)
        nn_idx_slice = d2_slice.argmin(dim=1)
        
        # Construct ground-truth direction vectors
        X_nn_slice = X_surf[nn_idx_slice]
        dir_gt_slice = X_slice_safe - X_nn_slice
        dir_gt_slice_norm = dir_gt_slice / (dir_gt_slice.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute projection points
        space_pred = sdf_pred[grad_idx]
        X_proj = X_slice_safe - dir_gt_slice_norm * space_pred.abs().unsqueeze(1)
        
        # Compute minimum distance from projection points to surface
        dists = torch.cdist(X_proj, X_surf).min(dim=1)[0]
        projection_loss = dists.abs().mean()
    
    # Reshape results back to grid shape for visualization
    projection_loss_grid = torch.zeros(resolution, resolution, device=device)
    
    # Create a mask to mark which points have valid projection loss
    valid_mask = torch.zeros(X_slice.shape[0], device=device, dtype=torch.bool)
    valid_mask[grad_idx] = True
    valid_mask_grid = valid_mask.view(resolution, resolution)
    
    # Fill projection loss into grid
    projection_loss_values = torch.zeros(X_slice.shape[0], device=device)
    projection_loss_values[grad_idx] = dists.abs()
    projection_loss_grid = projection_loss_values.view(resolution, resolution)
    
    # Set invalid points to NaN
    projection_loss_grid[~valid_mask_grid] = float('nan')
    
    return projection_loss_grid.cpu().numpy(), projection_loss.item(), slice_name


def visualize_projection_loss_slices(checkpoint_path, config_path, mesh_path='data/armadillo.obj', 
                                   resolution=128, save_path=None):
    """Visualize projection loss of middle slices in xyz three directions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define slices
    slices_info = [
        (0, 0.0, "YZ Plane (X=0.0)"),  # YZ plane
        (1, 0.0, "XZ Plane (Y=0.0)"),  # XZ plane  
        (2, 0.0, "XY Plane (Z=0.0)")   # XY plane
    ]
    
    projection_losses = []
    
    for i, (axis, pos, title) in enumerate(slices_info):
        print(f"Computing projection loss for {title}...")
        
        # Compute projection loss
        loss_grid, avg_loss, slice_name = compute_projection_loss_slice(
            checkpoint_path, config_path, mesh_path, axis, pos, resolution
        )
        projection_losses.append(avg_loss)
        
        # First row: projection loss heatmap
        im1 = axes[0, i].imshow(
            loss_grid.T, 
            origin='lower',
            extent=[-1, 1, -1, 1],
            cmap='hot',
            interpolation='nearest'
        )
        axes[0, i].set_title(f'{title}\nProjection Loss (Average: {avg_loss:.6f})')
        
        if axis == 0:  # YZ plane
            axes[0, i].set_xlabel('Y')
            axes[0, i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Z')
        else:  # XY plane
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Y')
            
        fig.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Second row: log scale visualization of projection loss (better for showing details)
        loss_grid_log = np.log10(loss_grid + 1e-8)  # Avoid log(0)
        im2 = axes[1, i].imshow(
            loss_grid_log.T,
            origin='lower', 
            extent=[-1, 1, -1, 1],
            cmap='viridis',
            interpolation='nearest'
        )
        axes[1, i].set_title(f'{title}\nLog10(Projection Loss)')
        
        if axis == 0:  # YZ plane
            axes[1, i].set_xlabel('Y')
            axes[1, i].set_ylabel('Z')
        elif axis == 1:  # XZ plane
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Z')
        else:  # XY plane
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Y')
            
        fig.colorbar(im2, ax=axes[1, i], shrink=0.8)
    
    plt.tight_layout()
    
    # Print summary
    print("\n=== Projection Loss Summary ===")
    for i, (axis, pos, title) in enumerate(slices_info):
        print(f"{title}: {projection_losses[i]:.6f}")
    print(f"Overall Average Projection Loss: {np.mean(projection_losses):.6f}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization results saved to: {save_path}")
    else:
        plt.show()
    
    return projection_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径') 
    parser.add_argument('--mesh', type=str, default='data/armadillo.obj', help='mesh文件路径')
    parser.add_argument('--resolution', type=int, default=128, help='切片分辨率')
    parser.add_argument('--save_path', type=str, default='comparative_experiment/projection_loss_slices.png', 
                       help='保存路径')
    
    args = parser.parse_args()
    
    print(f"加载模型: {args.checkpoint}")
    print(f"配置文件: {args.config}")
    print(f"Mesh文件: {args.mesh}")
    print(f"分辨率: {args.resolution}")
    
    # 计算并可视化projection loss
    projection_losses = visualize_projection_loss_slices(
        args.checkpoint, 
        args.config, 
        args.mesh,
        args.resolution,
        args.save_path
    )


if __name__ == "__main__":
    main() 