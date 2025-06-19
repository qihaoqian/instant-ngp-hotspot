#!/usr/bin/env python3
"""
运行projection loss计算的示例脚本
使用方法:
python comparative_experiment/run_projection_loss_example.py

你需要修改以下路径为你的实际文件路径:
- CHECKPOINT_PATH: 训练好的模型checkpoint路径
- CONFIG_PATH: 配置文件路径  
- MESH_PATH: mesh文件路径
"""

import os
from compute_projection_loss import visualize_projection_loss_slices

def main():
    # 设置路径 - 请根据你的实际情况修改这些路径
    CHECKPOINT_PATH = "workspace/ngp_armadillo/checkpoints/ngp_ep1000.pth"  # 修改为你的模型路径
    CONFIG_PATH = "config/armadillo.json"  # 修改为你的配置文件路径
    MESH_PATH = "data/armadillo.obj"  # 修改为你的mesh文件路径
    
    # 检查文件是否存在
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 找不到模型文件 {CHECKPOINT_PATH}")
        print("请修改 CHECKPOINT_PATH 为你的实际模型文件路径")
        return
        
    if not os.path.exists(CONFIG_PATH):
        print(f"错误: 找不到配置文件 {CONFIG_PATH}")
        print("请修改 CONFIG_PATH 为你的实际配置文件路径")
        return
        
    if not os.path.exists(MESH_PATH):
        print(f"错误: 找不到mesh文件 {MESH_PATH}")
        print("请修改 MESH_PATH 为你的实际mesh文件路径")
        return
    
    print("开始计算projection loss...")
    print(f"模型: {CHECKPOINT_PATH}")
    print(f"配置: {CONFIG_PATH}")
    print(f"Mesh: {MESH_PATH}")
    print("=" * 50)
    
    # 计算并可视化projection loss
    projection_losses = visualize_projection_loss_slices(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        mesh_path=MESH_PATH,
        resolution=128,  # 可以调整分辨率，值越大计算越慢但更精确
        save_path="comparative_experiment/projection_loss_slices.png"
    )
    
    print("=" * 50)
    print("计算完成！")
    print(f"可视化结果已保存至: comparative_experiment/projection_loss_slices.png")

if __name__ == "__main__":
    main() 