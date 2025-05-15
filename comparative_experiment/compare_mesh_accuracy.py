import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import argparse

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

def main(gt_path, pred_path):
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
    # Write results to file
    with open("comparative_experiment/mesh_accuracy_results.txt", "a") as f:
        f.write(f"Predicted Mesh: {pred_path}\n")
        f.write(f"Chamfer Distance: {chamfer:.6f}\n")
        f.write(f"Hausdorff Distance: {hausdorff:.6f}\n")
        f.write("-" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two meshes for reconstruction accuracy.")
    parser.add_argument("--gt", type=str, default="data/armadillo.obj", help="Path to ground truth mesh file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted mesh file.")
    args = parser.parse_args()

    main(args.gt, args.pred)
