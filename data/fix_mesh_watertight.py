import trimesh
import sys
import os
import numpy as np
from pathlib import Path


def fix_mesh_watertight(mesh_path, output_path=None, method='auto'):
    """
    修复mesh使其watertight
    
    Args:
        mesh_path (str): 输入mesh文件的路径
        output_path (str): 输出文件路径，如果为None则在原文件名基础上添加_fixed后缀
        method (str): 修复方法 ('auto', 'fill_holes', 'convex_hull', 'voxel')
        
    Returns:
        str: 修复后的文件路径
    """
    try:
        # 加载mesh
        print(f"加载mesh文件: {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # 检查原始状态
        print(f"原始mesh是否watertight: {mesh.is_watertight}")
        print(f"原始mesh顶点数: {len(mesh.vertices)}")
        print(f"原始mesh面数: {len(mesh.faces)}")
        
        if mesh.is_watertight:
            print("Mesh已经是watertight，无需修复")
            return mesh_path
        
        # 创建mesh的副本进行修复
        fixed_mesh = mesh.copy()
        
        if method == 'auto' or method == 'fill_holes':
            print("\n尝试填充孔洞修复...")
            # 基础清理
            fixed_mesh.remove_degenerate_faces()
            fixed_mesh.remove_duplicate_faces()
            fixed_mesh.remove_infinite_values()
            
            # 填充孔洞
            try:
                fixed_mesh.fill_holes()
                print("孔洞填充完成")
            except Exception as e:
                print(f"孔洞填充失败: {e}")
            
            # 修复法线
            try:
                fixed_mesh.fix_normals()
                print("法线修复完成")
            except Exception as e:
                print(f"法线修复失败: {e}")
            
            # 检查是否修复成功
            if fixed_mesh.is_watertight:
                print("✓ 通过填充孔洞成功修复为watertight")
            elif method == 'auto':
                print("填充孔洞方法失败，尝试凸包方法...")
                method = 'convex_hull'
            else:
                print("✗ 填充孔洞方法修复失败")
                
        if method == 'convex_hull' and not fixed_mesh.is_watertight:
            print("\n使用凸包方法修复...")
            try:
                # 使用凸包
                convex_hull = mesh.convex_hull
                if convex_hull.is_watertight:
                    fixed_mesh = convex_hull
                    print("✓ 通过凸包成功修复为watertight")
                else:
                    print("✗ 凸包方法修复失败")
                    if method == 'auto':
                        method = 'voxel'
            except Exception as e:
                print(f"凸包方法失败: {e}")
                if method == 'auto':
                    method = 'voxel'
        
        if method == 'voxel' and not fixed_mesh.is_watertight:
            print("\n使用体素化方法修复...")
            try:
                # 体素化然后重建
                voxel_size = max(mesh.bounds[1] - mesh.bounds[0]) / 64
                voxelized = mesh.voxelized(pitch=voxel_size)
                fixed_mesh = voxelized.marching_cubes
                
                if fixed_mesh.is_watertight:
                    print("✓ 通过体素化成功修复为watertight")
                else:
                    print("✗ 体素化方法修复失败")
            except Exception as e:
                print(f"体素化方法失败: {e}")
        
        # 最终检查
        print(f"\n修复后mesh是否watertight: {fixed_mesh.is_watertight}")
        print(f"修复后mesh顶点数: {len(fixed_mesh.vertices)}")
        print(f"修复后mesh面数: {len(fixed_mesh.faces)}")
        
        # 确定输出路径
        if output_path is None:
            path = Path(mesh_path)
            output_path = str(path.parent / f"{path.stem}_fixed{path.suffix}")
        
        # 保存修复后的mesh
        fixed_mesh.export(output_path)
        print(f"修复后的mesh已保存到: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"修复过程中发生错误: {str(e)}")
        return None


def batch_fix_meshes(input_dir, output_dir=None, method='auto'):
    """
    批量修复目录中的所有mesh文件
    
    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录，如果为None则在输入目录下创建fixed子目录
        method (str): 修复方法
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "fixed"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # 支持的mesh文件格式
    extensions = ['.obj', '.ply', '.stl', '.off']
    
    mesh_files = []
    for ext in extensions:
        mesh_files.extend(input_path.glob(f"*{ext}"))
    
    print(f"找到 {len(mesh_files)} 个mesh文件")
    
    for mesh_file in mesh_files:
        print(f"\n处理: {mesh_file.name}")
        output_file = output_path / f"{mesh_file.stem}_fixed{mesh_file.suffix}"
        fix_mesh_watertight(str(mesh_file), str(output_file), method)


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  单个文件: python fix_mesh_watertight.py <mesh_file> [output_file] [method]")
        print("  批量处理: python fix_mesh_watertight.py --batch <input_dir> [output_dir] [method]")
        print("方法选项: auto, fill_holes, convex_hull, voxel")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("批量模式需要指定输入目录")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        method = sys.argv[4] if len(sys.argv) > 4 else 'auto'
        
        batch_fix_meshes(input_dir, output_dir, method)
    else:
        mesh_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        method = sys.argv[3] if len(sys.argv) > 3 else 'auto'
        
        if not os.path.exists(mesh_path):
            print(f"错误: 文件 {mesh_path} 不存在")
            sys.exit(1)
        
        fix_mesh_watertight(mesh_path, output_path, method)


if __name__ == "__main__":
    main() 