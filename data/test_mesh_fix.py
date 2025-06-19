#!/usr/bin/env python3
"""
测试mesh修复功能的示例脚本
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hotspot.dataset import SDFDataset

def create_mesh_check_dir():
    """创建mesh_check目录"""
    check_dir = "mesh_check"
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
        print(f"创建目录: {check_dir}")
    return check_dir

def visualize_sdf_slices(dataset, mesh_path):
    """
    可视化3个方向的SDF切片（原始+二值化）并拼接成一张图片
    
    Args:
        dataset: SDFDataset实例
        mesh_path: mesh文件路径，用于生成输出文件名
    """
    try:
        # 创建输出目录
        check_dir = create_mesh_check_dir()
        
        # 获取mesh文件名（不包含路径和扩展名）
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        
        print(f"正在生成SDF切片可视化...")
        
        # 使用dataset自带的方法生成6张图的拼接（3个方向 × 2种模式）
        print("  - 生成六张切片图（原始SDF + 二值化SDF）...")
        dataset.plot_all_sdf_combined(
            workspace=check_dir,
            coord=0.0,
            show_contour=True  # 在二值化图中显示物体表面等值线
        )
        
        # 将生成的文件重命名为我们期望的格式
        src_path = os.path.join(check_dir, "sdf_all_slices_combined.png")
        dst_path = os.path.join(check_dir, f"{mesh_name}_sdf_combined.png")
        
        if os.path.exists(src_path):
            import shutil
            shutil.move(src_path, dst_path)
            print(f"✓ 六张拼接图片已保存: {dst_path}")
        else:
            print("✗ 切片图片生成失败")
            
    except Exception as e:
        print(f"可视化过程出错: {e}")
        import traceback
        traceback.print_exc()

def combine_sdf_images_with_binary(original_images, binary_images, output_path, mesh_name):
    """
    将三个方向的SDF切片图（原始+二值化）拼接成一张图片
    
    Args:
        original_images: 三个原始图片文件的路径列表 [x, y, z]
        binary_images: 三个二值化图片文件的路径列表 [x, y, z]
        output_path: 输出图片路径
        mesh_name: mesh名称，用于标题
        
    Returns:
        输出图片路径
    """
    try:
        # 读取所有图片
        original_imgs = [Image.open(path) for path in original_images]
        binary_imgs = [Image.open(path) for path in binary_images]
        
        # 获取图片尺寸
        img_width, img_height = original_imgs[0].size
        
        # 创建拼接图 (2行3列)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SDF Visualization - {mesh_name}', fontsize=16, weight='bold', y=0.95)
        
        axes_labels = ['X-slice (YZ)', 'Y-slice (XZ)', 'Z-slice (XY)']
        row_labels = ['Original SDF', 'Binary SDF (Inside/Outside)']
        
        # 显示图片
        for row, (imgs, row_label) in enumerate(zip([original_imgs, binary_imgs], row_labels)):
            for col, (img, axis_label) in enumerate(zip(imgs, axes_labels)):
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_title(f'{axis_label}', fontsize=10)
                ax.axis('off')
        
        # 在左侧添加行标签
        for row, label in enumerate(row_labels):
            fig.text(0.02, 0.75 - row * 0.45, label, rotation=90, 
                    fontsize=12, weight='bold', ha='center', va='center')
        
        # 添加颜色说明
        fig.text(0.5, 0.02, 'Original: Blue=Inside (SDF<0), Red=Outside (SDF>0) | Binary: Inside vs Outside', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.9, bottom=0.1)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"六张图片拼接失败: {e}")
        return None

def combine_slice_images(image_paths, output_path, mesh_name):
    """
    将三个方向的SDF切片图拼接成一张图片（兼容性保留）
    
    Args:
        image_paths: 三个图片文件的路径列表 [x, y, z]
        output_path: 输出图片路径
        mesh_name: mesh名称，用于标题
        
    Returns:
        输出图片路径
    """
    try:
        # 读取三张图片
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            images.append(img)
        
        # 获取图片尺寸
        img_width, img_height = images[0].size
        
        # 创建拼接后的图片 (3列1行布局)
        combined_width = img_width * 3
        combined_height = img_height + 60  # 额外空间用于标题
        
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        
        # 拼接图片
        axes_labels = ['X-slice (YZ plane)', 'Y-slice (XZ plane)', 'Z-slice (XY plane)']
        for i, (img, label) in enumerate(zip(images, axes_labels)):
            # 粘贴图片
            x_offset = i * img_width
            combined_img.paste(img, (x_offset, 60))  # 留出顶部空间给标题
        
        # 添加标题和轴标签
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # 使用matplotlib来添加文字
        fig, ax = plt.subplots(figsize=(combined_width/100, combined_height/100), dpi=100)
        ax.imshow(np.array(combined_img))
        
        # 添加主标题
        ax.text(combined_width/2, 30, f'SDF Ground Truth Visualization - {mesh_name}', 
                fontsize=16, ha='center', va='center', weight='bold')
        
        # 添加子标题
        for i, label in enumerate(axes_labels):
            x_center = (i + 0.5) * img_width
            ax.text(x_center, 45, label, fontsize=12, ha='center', va='center')
        
        ax.set_xlim(0, combined_width)
        ax.set_ylim(combined_height, 0)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"图片拼接失败: {e}")
        # 如果拼接失败，至少保存原始图片
        return image_paths[0] if image_paths else None

def visualize_sdf_comparison(dataset_original, dataset_fixed, mesh_path):
    """
    可视化修复前后的SDF对比
    
    Args:
        dataset_original: 原始dataset
        dataset_fixed: 修复后的dataset  
        mesh_path: mesh文件路径
    """
    try:
        check_dir = create_mesh_check_dir()
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        
        print("  - 生成修复前后SDF对比图...")
        
        # 为两个dataset分别生成6张图的拼接
        datasets = [dataset_original, dataset_fixed]
        labels = ['original', 'fixed']
        combined_images = []
        
        for dataset, label in zip(datasets, labels):
            # 为每个dataset创建临时目录
            temp_workspace = os.path.join(check_dir, f"temp_{label}")
            os.makedirs(temp_workspace, exist_ok=True)
            
            # 生成6张图的拼接
            dataset.plot_all_sdf_combined(
                workspace=temp_workspace,
                coord=0.0,
                show_contour=True
            )
            
            # 检查生成的文件
            combined_img_path = os.path.join(temp_workspace, "sdf_all_slices_combined.png")
            if os.path.exists(combined_img_path):
                combined_images.append(combined_img_path)
        
        # 创建对比图：将两个6张图的拼接图垂直排列
        if len(combined_images) == 2:
            comparison_path = os.path.join(check_dir, f"{mesh_name}_sdf_comparison.png")
            create_comparison_image_with_combined(combined_images, comparison_path, mesh_name, labels)
            print(f"✓ 对比图已保存: {comparison_path}")
        
        # 清理临时文件
        for label in labels:
            temp_dir = os.path.join(check_dir, f"temp_{label}")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"对比可视化出错: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_image(all_images, output_path, mesh_name):
    """
    创建修复前后的对比图像
    
    Args:
        all_images: [[original_x, original_y, original_z], [fixed_x, fixed_y, fixed_z]]
        output_path: 输出路径
        mesh_name: mesh名称
    """
    try:
        # 读取所有图片
        original_imgs = [Image.open(path) for path in all_images[0]]
        fixed_imgs = [Image.open(path) for path in all_images[1]]
        
        # 获取图片尺寸
        img_width, img_height = original_imgs[0].size
        
        # 创建对比图 (2行3列)
        combined_width = img_width * 3
        combined_height = img_height * 2 + 100  # 额外空间用于标题
        
        # 使用matplotlib创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SDF Comparison: {mesh_name}', fontsize=16, weight='bold', y=0.95)
        
        axes_labels = ['X-slice (YZ)', 'Y-slice (XZ)', 'Z-slice (XY)']
        row_labels = ['Original (Non-watertight)', 'Fixed (Watertight)']
        
        # 显示图片
        for row, (imgs, row_label) in enumerate(zip([original_imgs, fixed_imgs], row_labels)):
            for col, (img, axis_label) in enumerate(zip(imgs, axes_labels)):
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_title(f'{axis_label}' + (f'\n{row_label}' if col == 1 else ''), 
                           fontsize=10, weight='bold' if col == 1 else 'normal')
                ax.axis('off')
        
        # 在左侧添加行标签
        for row, label in enumerate(row_labels):
            fig.text(0.02, 0.75 - row * 0.45, label, rotation=90, 
                    fontsize=12, weight='bold', ha='center', va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.9)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"创建对比图失败: {e}")
        return None

def create_comparison_image_with_combined(combined_images, output_path, mesh_name, labels):
    """
    创建修复前后的对比图像（使用已拼接的6图组合）
    
    Args:
        combined_images: [original_combined_path, fixed_combined_path]
        output_path: 输出路径
        mesh_name: mesh名称
        labels: ['original', 'fixed']
    """
    try:
        # 读取两张拼接图片
        original_img = Image.open(combined_images[0])
        fixed_img = Image.open(combined_images[1])
        
        # 创建对比图 (2行1列)
        fig, axes = plt.subplots(2, 1, figsize=(18, 24))
        fig.suptitle(f'SDF Comparison (Original + Binary): {mesh_name}', fontsize=20, weight='bold', y=0.98)
        
        row_labels = ['Original (Non-watertight)', 'Fixed (Watertight)']
        
        # 显示图片
        for row, (img, row_label) in enumerate(zip([original_img, fixed_img], row_labels)):
            axes[row].imshow(img)
            axes[row].set_title(f'{row_label}', fontsize=16, weight='bold', pad=20)
            axes[row].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"创建对比图失败: {e}")
        return None

def test_mesh_fix(mesh_path, test_auto_fix=True):
    """
    测试mesh修复功能
    
    Args:
        mesh_path: mesh文件路径
        test_auto_fix: 是否测试自动修复功能
    """
    print(f"测试mesh文件: {mesh_path}")
    
    # 测试原始mesh（不自动修复）
    print("\n=== 加载原始mesh（不自动修复）===")
    try:
        dataset_no_fix = SDFDataset(
            path=mesh_path,
            size=1,
            num_samples_surf=1000,
            num_samples_space=1000,
            auto_fix_watertight=False
        )
        print(f"原始mesh是否watertight: {dataset_no_fix.mesh.is_watertight}")
        
        # 如果不测试自动修复，则只可视化原始mesh
        if not test_auto_fix:
            print("\n=== 测试数据采样（原始mesh）===")
            sample = dataset_no_fix[0]
            print(f"表面点数量: {sample['points_surf'].shape[0]}")
            print(f"内部点数量: {sample['points_occupied'].shape[0]}")
            print(f"外部点数量: {sample['points_free'].shape[0]}")
            
            # 检查SDF符号
            surf_sdf = sample['sdfs_surf']
            occupied_sdf = sample['sdfs_occupied']
            free_sdf = sample['sdfs_free']
            
            print(f"\n表面SDF值范围: [{surf_sdf.min():.4f}, {surf_sdf.max():.4f}]")
            print(f"内部SDF值范围: [{occupied_sdf.min():.4f}, {occupied_sdf.max():.4f}]")
            print(f"外部SDF值范围: [{free_sdf.min():.4f}, {free_sdf.max():.4f}]")
            
            # 检查符号正确性
            correct_surf = (surf_sdf <= 0.1).all()  # 表面点SDF应该接近0
            correct_occupied = (occupied_sdf < 0).all()  # 内部点SDF应该为负
            correct_free = (free_sdf > 0).all()  # 外部点SDF应该为正
            
            print(f"\nSDF符号检查:")
            print(f"表面点SDF符号正确: {correct_surf}")
            print(f"内部点SDF符号正确: {correct_occupied}")
            print(f"外部点SDF符号正确: {correct_free}")
            
            if correct_occupied and correct_free:
                print("✓ SDF符号计算正确!")
            else:
                print("✗ SDF符号可能有问题")
            
            # 生成原始mesh的SDF可视化
            print("\n=== 生成原始mesh SDF可视化 ===")
            visualize_sdf_slices(dataset_no_fix, mesh_path)
            return
            
    except Exception as e:
        print(f"加载原始mesh失败: {e}")
        return

    # 如果要测试自动修复
    if test_auto_fix:
        print("\n=== 自动修复测试 ===")
        try:
            dataset_with_fix = SDFDataset(
                path=mesh_path,
                size=1,
                num_samples_surf=1000,
                num_samples_space=1000,
                auto_fix_watertight=True
            )
            print(f"修复后mesh是否watertight: {dataset_with_fix.mesh.is_watertight}")
            
            # 测试数据采样
            print("\n=== 测试数据采样（修复后）===")
            sample = dataset_with_fix[0]
            print(f"表面点数量: {sample['points_surf'].shape[0]}")
            print(f"内部点数量: {sample['points_occupied'].shape[0]}")
            print(f"外部点数量: {sample['points_free'].shape[0]}")
            
            # 检查SDF符号
            surf_sdf = sample['sdfs_surf']
            occupied_sdf = sample['sdfs_occupied']
            free_sdf = sample['sdfs_free']
            
            print(f"\n表面SDF值范围: [{surf_sdf.min():.4f}, {surf_sdf.max():.4f}]")
            print(f"内部SDF值范围: [{occupied_sdf.min():.4f}, {occupied_sdf.max():.4f}]")
            print(f"外部SDF值范围: [{free_sdf.min():.4f}, {free_sdf.max():.4f}]")
            
            # 检查符号正确性
            correct_surf = (surf_sdf <= 0.1).all()  # 表面点SDF应该接近0
            correct_occupied = (occupied_sdf < 0).all()  # 内部点SDF应该为负
            correct_free = (free_sdf > 0).all()  # 外部点SDF应该为正
            
            print(f"\nSDF符号检查:")
            print(f"表面点SDF符号正确: {correct_surf}")
            print(f"内部点SDF符号正确: {correct_occupied}")
            print(f"外部点SDF符号正确: {correct_free}")
            
            if correct_occupied and correct_free:
                print("✓ SDF符号计算正确!")
            else:
                print("✗ SDF符号可能有问题")
                
            # 生成SDF可视化
            print("\n=== 生成SDF可视化 ===")
            
            # 如果原始mesh不是watertight，生成对比图
            if not dataset_no_fix.mesh.is_watertight and dataset_with_fix.mesh.is_watertight:
                print("生成修复前后对比图...")
                visualize_sdf_comparison(dataset_no_fix, dataset_with_fix, mesh_path)
            else:
                print("生成修复后的SDF可视化...")
                visualize_sdf_slices(dataset_with_fix, mesh_path)
                
        except Exception as e:
            print(f"自动修复测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="测试mesh修复功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
1. 检查mesh是否watertight
2. 可选择是否测试自动修复功能
3. 测试SDF计算的正确性
4. 生成SDF ground truth可视化
5. 保存结果到mesh_check文件夹

示例:
  python test_mesh_fix.py data/armadillo.obj                    # 仅测试原始mesh（默认）
  python test_mesh_fix.py data/armadillo.obj --no-auto-fix      # 明确仅测试原始mesh
  python test_mesh_fix.py data/armadillo.obj --auto-fix         # 测试自动修复功能
        """
    )
    
    parser.add_argument('mesh_path', help='mesh文件路径')
    
    # 互斥的参数组：auto-fix 和 no-auto-fix
    fix_group = parser.add_mutually_exclusive_group()
    fix_group.add_argument('--auto-fix', action='store_true',
                          help='测试自动修复功能')
    fix_group.add_argument('--no-auto-fix', action='store_true', default=True,
                          help='仅测试原始mesh，不进行自动修复（默认）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.mesh_path):
        print(f"错误: 文件 {args.mesh_path} 不存在")
        sys.exit(1)
    
    # 确定是否测试自动修复
    test_auto_fix = args.auto_fix
    
    test_mesh_fix(args.mesh_path, test_auto_fix)

if __name__ == "__main__":
    main() 