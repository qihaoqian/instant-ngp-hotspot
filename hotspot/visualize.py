import matplotlib.pyplot as plt
import numpy as np
from hotspot.utils import extract_fields

def get_sdfs_cross_section(bound_min, bound_max, resolution, query_func):
    """
    从sdfs的三维网格数据中提取xz平面的切面并进行可视化调试。

    参数:
        bound_min: 三元组 (x_min, y_min, z_min)，表示网格区域的最小坐标。
        bound_max: 三元组 (x_max, y_max, z_max)，表示网格区域的最大坐标。
        resolution: 三元组 (nx, ny, nz)，表示在每个方向上的分辨率。
        query_func: 用于计算每个点 sdf 值的函数（会传给 extract_fields 使用）。

    返回:
        cross_section: xz平面的二维 numpy 数组，形状为 (nx, nz)。
    """
    # 提取整个网格的 sdf 数值，假设 extract_fields 已实现
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    
    # 假设 u 的形状为 (nx, ny, nz)，我们选择中间的 y 层作为切面
    ny = u.shape[1]
    y_index = ny // 2  # 选择中间层
    cross_section = u[:, y_index, :]  # 结果形状为 (nx, nz)
    
    # 计算x和z方向上的物理坐标范围，用于图像显示的extent参数
    x_min, y_min, z_min = bound_min
    x_max, y_max, z_max = bound_max
    extent = [x_min, x_max, z_min, z_max]
    
    # 可视化切面，使用颜色映射来区分不同的sdf值
    plt.figure(figsize=(8, 6))
    # 转置切面以匹配x轴为水平，z轴为垂直，并设置 origin='lower'
    plt.imshow(cross_section.T, origin='lower', extent=extent, cmap='jet')
    plt.colorbar(label='sdfs 距离')
    plt.xlabel('x')
    plt.ylabel('z')
    # 使用中间的y值作为切面说明
    y_value = y_min + (y_max - y_min) / 2.0
    plt.title(f'SDFs xz切面 (y = {y_value:.2f})')
    plt.savefig('sdfs_cross_section.png', dpi=300, bbox_inches='tight')
    plt.show()