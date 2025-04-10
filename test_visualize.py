import trimesh
import numpy as np
import matplotlib.pyplot as plt

def visualize_cross_section(mesh, plane_normal=np.array([0, 0, 1]), plane_origin=None):
    """
    提取并可视化 mesh 的一个切面，例如取 xy 平面（z 轴为法向量）在 z 轴中间的截面
    
    参数：
      mesh: trimesh.Trimesh 对象
      plane_normal: 平面法向量，默认是 y 轴方向 [0,1,0]，代表 xz 平面
      plane_origin: 平面上的一点，如果为 None，则默认取 mesh 的包围盒中心（即 z 坐标的中间值）
    """
    # 如果未指定切面原点，则使用 mesh 的包围盒中心
    if plane_origin is None:
        plane_origin = mesh.bounds.mean(axis=0)
    
    # 计算与平面的交线（切面），这里 section 方法返回的是二维路径
    slice_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if slice_section is None:
        print("没有在指定的平面上找到截面")
        return
    
    if slice_section is not None:
        # 转换为2D平面截面
        slice_2D, _ = slice_section.to_planar()

        # 绘制截面线
        fig, ax = plt.subplots(figsize=(8, 8))
        for entity in slice_2D.entities:
            points = slice_2D.vertices[entity.points]
            ax.plot(points[:, 0], points[:, 1], 'k-')

        # 保持比例一致
        ax.axis('equal')

        # 保存为图片
        plt.savefig('section.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    
if __name__ == '__main__':
    mesh = trimesh.load('data/armadillo.obj', force='mesh')
    visualize_cross_section(mesh, plane_normal=np.array([0, 0, 1]), plane_origin=None)