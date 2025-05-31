import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import argparse
import trimesh
import pysdf
import os, sys
# 把 project_root 加到 import 搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import Config
from hotspot.network import SDFNetwork
from hotspot.utils import *
import torch

def sample_points_from_mesh(mesh_path, num_points=100000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(sampled.points)

def chamfer_distance(pcd1, pcd2):
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)

    dist1, _ = tree1.query(pcd2)
    dist2, _ = tree2.query(pcd1)

    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer

def hausdorff_distance(pcd1, pcd2):
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)

    dist1, _ = tree1.query(pcd2)
    dist2, _ = tree2.query(pcd1)

    return max(np.max(dist1), np.max(dist2))

def ply_to_checkpoint(ply_path):
    # 拆分路径
    parts = ply_path.split(os.sep)
    
    # 替换路径中的 "validation" → "checkpoints"
    parts[parts.index("validation")] = "checkpoints"

    # 获取 ply 文件名中的数字
    ply_filename = os.path.splitext(parts[-1])[0]  # "ngp_123"
    number = int(ply_filename.split("_")[1])       # 123 → int
    
    # 构造新的文件名
    ckpt_filename = f"ngp_ep{number:04d}.pth"

    # 替换文件名
    parts[-1] = ckpt_filename

    # 重新组装路径
    checkpoint_path = os.path.join(*parts)
    return checkpoint_path

def visualize_sdf_slices(sdf_grid, save_path=None):
    """
    将三维 SDF 网格可视化为三个正交平面的中间切片：XY、XZ、YZ。
    
    参数:
        sdf_grid: ndarray, shape=(N,N,N)，signed distance field。
    """
    N = sdf_grid.shape[0]
    mid = N // 2

    # 准备 3 个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY 平面 (固定 Z = mid)
    im0 = axes[0].imshow(
        sdf_grid[:, :, mid].T, 
        origin='lower', 
        extent=[-1,1,-1,1]
    )
    axes[0].set_title(f'XY Slice (Z={mid})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # XZ 平面 (固定 Y = mid)
    im1 = axes[1].imshow(
        sdf_grid[:, mid, :].T, 
        origin='lower',
        extent=[-1,1,-1,1]
    )
    axes[1].set_title(f'XZ Slice (Y={mid})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # YZ 平面 (固定 X = mid)
    im2 = axes[2].imshow(
        sdf_grid[mid, :, :].T, 
        origin='lower',
        extent=[-1,1,-1,1]
    )
    axes[2].set_title(f'YZ Slice (X={mid})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300) if save_path else plt.show()

def load_model(checkpoint_path, config_path):
    #load model
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
    # print(model)
    # 1. 把模型搬到你想用的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 2. 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    #    —— 如果你保存的时候是：
    #       torch.save({'model_state_dict': model.state_dict(), ...}, path)
    model.load_state_dict(ckpt['model'])
    return model.eval(), device
    
def sdf_value_loss(checkpoint_path, config_path, mesh_path='data/armadillo.obj', resolution=256):
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    min_bound = mesh.bounds[0]  # Min corner of the bounding box
    max_bound = mesh.bounds[1]  # Max corner of the bounding box 
    center = (min_bound + max_bound) / 2
    bbox_diagonal = np.linalg.norm(max_bound - min_bound)  # Diagonal length of the bounding box
    scale_factor = 0.7 * np.sqrt(3) / (bbox_diagonal / 2)  # Scale so that the object's diagonal fits within the sphere
    mesh.vertices -= center  # Translate mesh to the origin
    mesh.vertices *= scale_factor  # Scale the mesh
    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
    
    # 3. 在 [-1,1]^3 上生成均匀网格
    xs = np.linspace(-1, 1, resolution, dtype=np.float32)
    ys = np.linspace(-1, 1, resolution, dtype=np.float32)
    zs = np.linspace(-1, 1, resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    # 4. 展平所有点，批量计算 signed distance
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # (N,3) where N = res^3
    # pysdf 返回 unsigned distance，若需 signed distance 可调用 signed_distance
    # 下面假设 pysdf.SDF 提供 signed_distance 接口
    sdf_vals_gt = -sdf_fn(pts)  # 返回形如 (N,) 的数组
    occ_idx = np.where(sdf_vals_gt < 0)[0]
    sdf_occ_gt = sdf_vals_gt[occ_idx]  # 只保留小于 0 的 SDF 值
    print(f"Number of points with sdf < 0: {len(occ_idx)} out of {pts.shape[0]}")

    # 5. 恢复原来的网格形状
    sdf_grid_gt_grid = sdf_vals_gt.reshape((resolution, resolution, resolution))
    visualize_sdf_slices(sdf_grid_gt_grid, save_path='comparative_experiment/sdf_slice_gt.png')
    
    # 6. 加载模型并预测 SDF
    model, device = load_model(checkpoint_path, config_path)
    pts_tensor = torch.from_numpy(pts).float().to(device)
    all_preds = []
    B = 200_000  # 每次 200K 点，按你显存调整
    with torch.no_grad():
        for i in range(0, pts.shape[0], B):
            chunk = pts_tensor[i : i + B]           # (B,3)
            pred_chunk = model(chunk)               # (B,1) 或 (B,)
            all_preds.append(pred_chunk.cpu())
    sdf_pred = torch.cat(all_preds, dim=0)
    sdf_pred_occ = sdf_pred[occ_idx].view(-1).cpu().numpy()
    sdf_pred = sdf_pred.view(-1).cpu().numpy()
    sdf_pred_grid = sdf_pred.reshape((resolution, resolution, resolution))
    visualize_sdf_slices(sdf_pred_grid, save_path='comparative_experiment/sdf_slice_pred.png')
    loss_sdf = np.mean(np.abs(sdf_pred - sdf_vals_gt))
    loss_sdf_occ = np.mean(np.abs(sdf_pred_occ - sdf_occ_gt))
    return loss_sdf, loss_sdf_occ

def main(gt_path, pred_path, config_path):
    print("Sampling points from ground truth mesh...")
    gt_points = sample_points_from_mesh(gt_path)
    print("Sampling points from predicted mesh...")
    pred_points = sample_points_from_mesh(pred_path)

    print("Computing Chamfer Distance...")
    chamfer = chamfer_distance(gt_points, pred_points)
    print(f"Chamfer Distance: {chamfer:.6f}")

    print("Computing Hausdorff Distance...")
    hausdorff = hausdorff_distance(gt_points, pred_points)
    print(f"Hausdorff Distance: {hausdorff:.6f}")
    
    print("Computing sdf value loss...")

    checkpoint_path = ply_to_checkpoint(pred_path)
    loss_sdf, loss_sdf_occ = sdf_value_loss(checkpoint_path, config_path, mesh_path=gt_path, resolution=256,)
    print(f"SDF Value Loss: {loss_sdf:.6f}")
    print(f"SDF Value Loss of occupied area: {loss_sdf_occ:.6f}")
    # Write results to file
    with open("comparative_experiment/mesh_accuracy_results.txt", "a") as f:
        f.write(f"Predicted Mesh: {pred_path}\n")
        f.write(f"Chamfer Distance: {chamfer:.6f}\n")
        f.write(f"Hausdorff Distance: {hausdorff:.6f}\n")
        f.write(f"SDF Value Loss: {loss_sdf:.6f}\n")
        f.write(f"SDF Value Loss of occupied area: {loss_sdf_occ:.6f}\n")
        f.write("-" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two meshes for reconstruction accuracy.")
    parser.add_argument("--gt", type=str, default="data/armadillo.obj", help="Path to ground truth mesh file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted mesh file.")
    parser.add_argument("--config", type=str, default=None, help="Path to model config file.")
    args = parser.parse_args()

    main(args.gt, args.pred, args.config)
