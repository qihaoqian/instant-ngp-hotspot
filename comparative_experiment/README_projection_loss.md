# Projection Loss 计算和可视化工具

这个工具仿照 `sdf_value_loss` 函数的实现方式，用于计算和可视化训练好的SDF模型在xyz三个方向切片上的 projection loss。

## 功能

- 加载训练好的SDF模型
- 在xyz三个方向(YZ、XZ、XY平面)上生成切片
- 计算每个切片点的projection loss
- 生成热力图可视化结果，包括:
  - 原始projection loss热力图
  - Log scale的projection loss热力图(便于观察细节)

## 文件说明

- `compute_projection_loss.py`: 主要脚本，包含所有计算和可视化功能
- `run_projection_loss_example.py`: 使用示例脚本
- `README_projection_loss.md`: 本说明文档

## 使用方法

### 方法1: 使用示例脚本 (推荐)

1. 编辑 `run_projection_loss_example.py` 中的路径:
   ```python
   CHECKPOINT_PATH = "你的模型checkpoint路径"
   CONFIG_PATH = "你的配置文件路径" 
   MESH_PATH = "你的mesh文件路径"
   ```

2. 运行脚本:
   ```bash
   cd comparative_experiment
   python run_projection_loss_example.py
   ```

### 方法2: 使用命令行参数

```bash
cd comparative_experiment
python compute_projection_loss.py \
    --checkpoint path/to/your/model.pth \
    --config path/to/your/config.json \
    --mesh path/to/your/mesh.obj \
    --resolution 128 \
    --save_path projection_loss_result.png
```

### 参数说明

- `--checkpoint`: 训练好的模型checkpoint文件路径(.pth)
- `--config`: 模型配置文件路径(.json)
- `--mesh`: Ground truth mesh文件路径(.obj)
- `--resolution`: 切片分辨率(默认128，值越大越精确但计算越慢)
- `--save_path`: 结果图片保存路径(默认为comparative_experiment/projection_loss_slices.png)

## Projection Loss 计算原理

Projection Loss 的计算步骤如下:

1. **加载mesh**: 加载ground truth mesh并进行标准化处理
2. **生成切片**: 在指定平面上生成规则网格点
3. **预测SDF**: 使用训练好的模型预测每个点的SDF值
4. **计算梯度**: 使用有限差分法计算SDF梯度
5. **找最近邻**: 为每个点找到mesh表面上最近的点
6. **计算投影**: 根据公式 `X_proj = X - dir_norm * |sdf_pred|` 计算投影点
7. **计算距离**: 计算投影点到mesh表面的最小距离作为projection loss

## 输出结果

运行后会生成一个包含6个子图的可视化结果:

**第一行**: 三个方向切片的projection loss热力图
- YZ平面 (固定X=0.0)
- XZ平面 (固定Y=0.0) 
- XY平面 (固定Z=0.0)

**第二行**: 对应的log scale热力图(更好地显示细节)

每个子图会显示:
- 切片方向和位置
- 平均projection loss数值
- 颜色条表示loss大小

## 注意事项

1. **内存使用**: 高分辨率(>256)可能需要大量显存，建议从128开始测试
2. **计算时间**: projection loss计算涉及最近邻搜索，较为耗时
3. **文件路径**: 确保所有文件路径正确，特别是相对路径的使用
4. **依赖项**: 需要安装 trimesh, torch, matplotlib, numpy 等依赖

## 故障排除

1. **找不到文件**: 检查路径是否正确，使用绝对路径避免相对路径问题
2. **显存不足**: 降低resolution参数或减少batch size
3. **计算很慢**: 这是正常的，projection loss计算本身较为复杂
4. **空白图像**: 检查模型加载是否成功，mesh文件是否正确

## 示例输出

运行成功后，终端会显示类似如下信息:

```
计算 YZ平面 (X=0.0) 的projection loss...
计算 XZ平面 (Y=0.0) 的projection loss...
计算 XY平面 (Z=0.0) 的projection loss...

=== Projection Loss 总结 ===
YZ平面 (X=0.0): 0.012345
XZ平面 (Y=0.0): 0.013456
XY平面 (Z=0.0): 0.011234
总体平均 Projection Loss: 0.012345

可视化结果已保存至: comparative_experiment/projection_loss_slices.png 