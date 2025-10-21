# 传统算法盲道检测工具包

## 概述

本项目提供了基于传统计算机视觉算法的盲道检测解决方案，包含以下三个主要工具：

1. **批量检测脚本** (`batch_blind_detection.py`) - 批量处理图片文件夹
2. **参数调优工具** (`parameter_tuning.py`) - 自动寻找最佳参数
3. **可视化调试工具** (`visual_debugger.py`) - 交互式调试和参数调整

## 目录结构

```
E:\Code\python\download\blind_road_dataset\
├── images\                    # 输入图片文件夹
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── results\                   # 输出结果文件夹（自动创建）
│   ├── image1_detected.jpg
│   ├── image2_detected.jpg
│   ├── detection_results.json
│   └── ...
├── batch_blind_detection.py   # 批量检测脚本
├── parameter_tuning.py        # 参数调优工具
├── visual_debugger.py         # 可视化调试工具
├── requirements.txt           # 依赖文件
├── best_params.json          # 最佳参数文件（自动生成）
└── README.md                 # 使用说明
```

## 环境要求

### 必需依赖
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tkinter  # 通常Python自带
```

### 一键安装
```bash
cd E:\Code\python\download\blind_road_dataset
pip install -r requirements.txt
```

## 使用步骤

### 第一步：准备数据

1. **创建目录结构**
   ```bash
   mkdir -p "E:\Code\python\download\blind_road_dataset\images"
   ```

2. **放入测试图片**
   - 将盲道图片放入 `E:\Code\python\download\blind_road_dataset\images` 文件夹
   - 支持的格式：JPG、JPEG、PNG、BMP、TIFF

### 第二步：参数调优（推荐）

1. **运行参数调优工具**
   ```bash
   cd E:\Code\python\download\blind_road_dataset
   python parameter_tuning.py
   ```

2. **等待调优完成**
   - 程序会自动测试多种参数组合
   - 找到最佳参数后保存到 `best_params.json`

3. **查看调优结果**
   - 程序会显示最佳参数组合
   - 参数文件可用于后续批量检测

### 第三步：批量检测

1. **使用默认参数检测**
   ```bash
   python batch_blind_detection.py
   ```

2. **查看检测结果**
   - 结果保存在 `results` 文件夹
   - 包含检测结果图片和JSON报告

### 第四步：可视化调试（可选）

1. **启动调试工具**
   ```bash
   python visual_debugger.py
   ```

2. **交互式调试**
   - 选择单张图片进行检测
   - 实时调整参数查看效果
   - 保存最佳参数设置

## 算法原理

### 检测流程

1. **图像预处理**
   - 灰度化：将彩色图片转换为灰度图
   - 去噪：使用高斯滤波去除噪声
   - 对比度增强：直方图均衡化

2. **边缘检测**
   - 使用Canny算法检测边缘
   - 自适应阈值计算

3. **直线检测**
   - 霍夫变换检测直线
   - 过滤短线段和噪声

4. **盲道识别**
   - 根据角度过滤（盲道通常是水平条纹）
   - 根据位置过滤（盲道通常在图像中心区域）
   - 根据长度过滤（盲道条纹有一定长度）

### 关键参数说明

| 参数 | 说明 | 推荐范围 |
|------|------|----------|
| `canny_ratio` | Canny边缘检测阈值比例 | 0.2-0.5 |
| `threshold` | 霍夫变换累加器阈值 | 30-100 |
| `minLineLength` | 最小线段长度 | 30-70 |
| `maxLineGap` | 最大线段间隙 | 5-15 |
| `max_angle` | 最大角度（度） | 20-45 |

## 输出结果

### 检测结果图片
- 文件名格式：`原文件名_detected.扩展名`
- 绿色线条：所有检测到的线条
- 红色线条：识别为盲道的线条

### JSON报告
```json
{
  "summary": {
    "total_images": 100,
    "detected_blind_path": 25,
    "detection_rate": 0.25,
    "error_count": 0,
    "timestamp": "2024-01-01T12:00:00"
  },
  "results": [
    {
      "image_path": "path/to/image.jpg",
      "total_lines": 15,
      "blind_lines": 3,
      "has_blind_path": true,
      "confidence": 0.3,
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

## 快速开始

### 完整流程示例
```bash
# 1. 进入项目目录
cd E:\Code\python\download\blind_road_dataset

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行参数调优
python parameter_tuning.py

# 4. 运行批量检测
python batch_blind_detection.py

# 5. 查看结果
dir results
```

### 单张图片测试
```python
from batch_blind_detection import BlindPathDetector

# 创建检测器
detector = BlindPathDetector()

# 检测单张图片
result = detector.detect_blind_path("images/test.jpg")
print(f"检测结果: {result}")
```

## 性能优化建议

### 1. 参数调优
- 使用 `parameter_tuning.py` 自动寻找最佳参数
- 针对特定场景调整参数范围
- 考虑图片质量和盲道特征

### 2. 图片预处理
- 确保图片清晰度足够
- 避免过暗或过亮的图片
- 考虑图片尺寸对检测效果的影响

### 3. 批量处理优化
- 对于大量图片，考虑分批处理
- 使用多进程加速（需要额外开发）
- 定期保存中间结果

## 常见问题

### Q1: 检测结果不准确怎么办？
**A:** 
1. 使用可视化调试工具调整参数
2. 检查图片质量和盲道特征
3. 重新运行参数调优

### Q2: 程序运行速度慢怎么办？
**A:**
1. 减少图片尺寸
2. 调整参数减少检测线条数量
3. 使用更快的硬件

### Q3: 出现错误怎么办？
**A:**
1. 检查图片文件是否损坏
2. 确认依赖库版本正确
3. 查看错误日志定位问题

### Q4: 如何提高检测准确率？
**A:**
1. 收集更多样化的训练图片
2. 针对特定场景优化参数
3. 考虑结合其他特征（颜色、纹理等）

## 扩展功能

### 1. 多进程处理
```python
import multiprocessing as mp

def process_image(args):
    image_path, params = args
    return detector.detect_blind_path(image_path)

# 使用多进程
with mp.Pool(processes=4) as pool:
    results = pool.map(process_image, image_params_list)
```

### 2. 实时检测
```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect_blind_path_frame(frame)
    cv2.imshow('Blind Path Detection', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 技术支持

如果遇到问题，请：
1. 查看错误日志
2. 检查环境配置
3. 参考常见问题解答
4. 使用可视化调试工具排查

## 更新日志

- **v1.0** - 初始版本，包含基本检测功能
- **v1.1** - 添加参数调优工具
- **v1.2** - 添加可视化调试工具
- **v1.3** - 优化检测算法，提高准确率
- **v1.4** - 更新路径配置，适配新的目录结构 