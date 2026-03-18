# 模型文件说明

## 模型文件夹结构

将下载的模型文件放在此 `models` 文件夹内。

## 需要的模型文件

### 1. 盲道障碍检测模型

**推荐模型：**
- `yolov8n.pt` - YOLOv8 Nano（轻量级，速度快）
- `yolov8s.pt` - YOLOv8 Small（平衡性能）
- `yolov8m.pt` - YOLOv8 Medium（较高精度）
- `yolov8l.pt` - YOLOv8 Large（高精度）
- `yolov8x.pt` - YOLOv8 XLarge（最高精度）

**自定义训练模型：**
- `blind_road_best.pt` - 训练后的盲道障碍检测模型
- `blind_road_detection.pt` - 盲道检测专用模型

### 2. 环境检测模型

**推荐模型：**
- `yolov8n.pt` - 可以复用（如果已下载）
- `environment_detection.pt` - 环境检测专用模型
- `environment_best.pt` - 训练后的环境检测模型

## 模型下载地址

### YOLOv8 预训练模型

**官方下载地址：**
1. GitHub Releases: https://github.com/ultralytics/assets/releases
2. 直接下载链接：
   - YOLOv8n: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   - YOLOv8s: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
   - YOLOv8m: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   - YOLOv8l: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
   - YOLOv8x: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

### 使用Python下载

```python
from ultralytics import YOLO

# 自动下载YOLOv8n模型
model = YOLO('yolov8n.pt')  # 首次使用会自动下载

# 或者手动下载
model = YOLO('yolov8n.pt')
model.export()  # 导出模型
```

### 使用命令行下载

```bash
# 使用wget (Linux/Mac)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# 使用curl (Linux/Mac)
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt

# 使用PowerShell (Windows)
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -OutFile models\yolov8n.pt
```

## 模型文件命名规范

为了便于系统识别模型类型，建议使用以下命名规范：

### 盲道障碍检测模型
- `blind_road_*.pt`
- `yolov8*.pt` (通用模型，可用于盲道检测)

### 环境检测模型
- `environment_*.pt`
- `env_*.pt`

## 模型放置位置

所有模型文件应放在：
```
blind_road_dataset/
  └── models/
      ├── yolov8n.pt
      ├── yolov8s.pt
      ├── blind_road_best.pt
      └── environment_best.pt
```

## 模型选择建议

### 开发/测试阶段
- 使用 `yolov8n.pt`（最小最快）
- 适合快速测试和原型开发

### 生产环境
- 使用训练后的自定义模型（如 `blind_road_best.pt`）
- 或使用 `yolov8m.pt` 或 `yolov8l.pt`（平衡精度和速度）

### 高精度需求
- 使用 `yolov8x.pt` 或训练后的高精度模型
- 注意：速度会较慢

## 注意事项

1. **模型文件大小：**
   - YOLOv8n: ~6MB
   - YOLOv8s: ~22MB
   - YOLOv8m: ~52MB
   - YOLOv8l: ~88MB
   - YOLOv8x: ~136MB

2. **首次使用：**
   - 如果模型文件不存在，程序会尝试自动下载
   - 需要网络连接

3. **训练后的模型：**
   - 训练完成后，模型会自动保存在 `results/` 或 `runs/detect/` 目录
   - 可以手动复制到 `models/` 文件夹以便统一管理

## 快速开始

1. 下载至少一个YOLOv8模型（推荐 `yolov8n.pt`）
2. 将模型文件放到 `models/` 文件夹
3. 启动程序，在模型测试界面选择模型进行测试
