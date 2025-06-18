import trimesh
import sys
import os
import numpy as np

def check_mesh_watertight(mesh_path):
    """
    检查mesh是否watertight
    
    Args:
        mesh_path (str): mesh文件的路径
        
    Returns:
        bool: 如果mesh是watertight返回True，否则返回False
    """
    try:
        # 加载mesh
        mesh = trimesh.load(mesh_path)
        
        # 检查是否watertight
        is_watertight = mesh.is_watertight
        
        # 打印详细信息
        print(f"\n检查文件: {os.path.basename(mesh_path)}")
        print(f"是否watertight: {is_watertight}")
        
        if not is_watertight:
            print("\n可能的问题:")
            
            # 检查法线是否一致
            print(f"- 法线是否一致: {mesh.is_winding_consistent}")
            
            # 检查面的数量
            print(f"- 面的数量: {len(mesh.faces)}")
            
            # 检查顶点的数量
            print(f"- 顶点的数量: {len(mesh.vertices)}")
            
            # 检查边界边的数量
            print(f"- 边界边的数量: {len(mesh.edges_unique[mesh.edges_sorted])}")
            
            # 检查是否有重复顶点
            unique_vertices = np.unique(mesh.vertices, axis=0)
            if len(unique_vertices) != len(mesh.vertices):
                print(f"- 重复顶点的数量: {len(mesh.vertices) - len(unique_vertices)}")
            
            # 提供修复建议
            print("\n建议的修复步骤:")
            print("1. mesh.fill_holes()  # 填充孔洞")
            print("2. mesh.remove_degenerate_faces()  # 移除退化的面")
            print("3. mesh.remove_duplicate_faces()  # 移除重复的面")
            print("4. mesh.remove_infinite_values()  # 移除无限值")
            print("5. mesh.fix_normals()  # 修复法线方向")
        
        return is_watertight
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("使用方法: python check_watertight.py <mesh_file_path>")
        sys.exit(1)
        
    mesh_path = sys.argv[1]
    if not os.path.exists(mesh_path):
        print(f"错误: 文件 {mesh_path} 不存在")
        sys.exit(1)
        
    check_mesh_watertight(mesh_path)

if __name__ == "__main__":
    main() 