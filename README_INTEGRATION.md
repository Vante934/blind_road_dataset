# 盲道检测系统 - 集成说明文档

本文档为B成员提供YOLO检测模块和数据库服务的使用说明，帮助您快速集成到前端应用中。

## 目录结构

```
your_project/
 ├── modules/                    # 核心功能模块
 │   ├── __init__.py
 │   ├── detector.py            # YOLO检测器封装
 │   ├── database_service.py    # 数据库服务封装
 │   └── voice_generator.py     # 语音文本生成
 │
 ├── models/
 │   ├── best.pt                # YOLO权重
 │   ├── database.py            # SQLAlchemy模型
 │   └── schemas.py             # 数据验证模型
 │
 ├── configs/
 │   └── config.json            # 配置文件
 │
 ├── tests/                      # 测试文件
 │   ├── test_detector.py
 │   └── test_database.py
 │
 ├── requirements_detection.txt  # 依赖清单
 └── README_INTEGRATION.md       # 使用说明
```

## 安装依赖

首先安装所需的依赖包：

```bash
pip install -r requirements_detection.txt
```

## YOLO检测模块使用说明

### 1. 初始化检测器

```python
from modules.detector import BlindRoadDetector, get_detector

# 方法1：直接初始化
model_path = "models/best.pt"
detector = BlindRoadDetector(model_path)

# 方法2：使用单例模式（推荐）
detector = get_detector(model_path)
```

### 2. 图像检测

```python
# 使用图像字节流（适用于FastAPI上传的文件）
with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()
detection_results = detector.detect_from_bytes(image_bytes)

# 使用图像数组（OpenCV格式）
import cv2
image = cv2.imread("path/to/image.jpg")
detection_results = detector.detect_from_frame(image)

# 设置置信度阈值
detection_results = detector.detect_from_bytes(image_bytes, conf_threshold=0.5)
```

### 3. 检测结果格式

检测结果是一个字典，包含以下字段：

```python
{
    "success": True,
    "obstacles": [
        {
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [0.1, 0.2, 0.3, 0.4],  # 归一化坐标 [x1, y1, x2, y2]
            "distance_estimate": 2.5,        # 估算距离（米）
            "direction": "center",          # 方向：left, center, right
            "danger_level": "medium"        # 危险等级：low, medium, high
        },
        # 更多障碍物...
    ],
    "blind_road_detected": True,    # 是否检测到盲道
    "blind_road_status": "on_track",  # 盲道状态：on_track, deviated_left, deviated_right, lost
    "voice_alert": "注意！前方2米处有行人",  # 语音提示
    "processing_time_ms": 45.2      # 处理时间（毫秒）
}
```

### 4. 批量检测

```python
# 批量处理多帧图像
import cv2
images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(1, 4)]
batch_results = detector.batch_detect(images)
```

## 数据库服务使用说明

### 1. 初始化数据库服务

```python
from modules.database_service import DatabaseService, get_database_service

# 方法1：直接初始化
db_service = DatabaseService()

# 方法2：使用单例模式（推荐）
db_service = get_database_service()

# 使用自定义数据库URL
database_url = "sqlite:///custom.db"
db_service = DatabaseService(database_url)
```

### 2. 用户管理

#### 创建用户

```python
user = db_service.create_user(
    username="john",
    email="john@example.com",
    password_hash="hashed_password"
)
print(f"创建用户成功，ID: {user.id}")
```

#### 获取用户

```python
# 根据用户名获取
user = db_service.get_user_by_username("john")

# 根据ID获取
user = db_service.get_user_by_id(1)

# 获取所有用户
users = db_service.get_all_users()
```

#### 更新用户

```python
updated_user = db_service.update_user(
    user_id=1,
    email="john.doe@example.com",
    is_active=True
)
```

#### 删除用户

```python
success = db_service.delete_user(user_id=1)
print(f"删除用户: {'成功' if success else '失败'}")
```

### 3. 检测记录管理

#### 保存检测记录

```python
# 准备检测结果
detection_result = [
    {
        'bbox': [100, 100, 200, 200],
        'class_id': 0,
        'class_name': 'person',
        'confidence': 0.95,
        'size': {'width': 100, 'height': 100}
    }
]

# 保存检测记录
detection = db_service.save_detection(
    user_id=1,
    image_path="path/to/image.jpg",
    detection_result=detection_result,
    detection_type="blind_road"
)
print(f"保存检测记录成功，ID: {detection.id}")
```

#### 获取检测记录

```python
# 获取用户的检测记录
user_detections = db_service.get_user_detections(user_id=1, limit=100)

# 根据ID获取检测记录
detection = db_service.get_detection_by_id(detection_id=1)
```

#### 获取检测统计信息

```python
stats = db_service.get_detection_statistics(user_id=1)
print(f"总检测次数: {stats['total_detections']}")
print(f"总目标数: {stats['total_targets']}")
print(f"平均置信度: {stats['avg_confidence']:.2f}")
print(f"检测类型分布: {stats['type_counts']}")
```

#### 删除检测记录

```python
success = db_service.delete_detection(detection_id=1)
print(f"删除检测记录: {'成功' if success else '失败'}")
```

### 4. 导航记录管理

#### 保存导航记录

```python
route_data = {
    'waypoints': [[39.9042, 116.4074], [39.9045, 116.4077]],
    'distance': 100.5,
    'duration': 120
}

record = db_service.save_navigation_record(
    user_id=1,
    start_latitude=39.9042,
    start_longitude=116.4074,
    end_latitude=39.9045,
    end_longitude=116.4077,
    route_data=route_data,
    distance=100.5,
    duration=120,
    navigation_mode="walking",
    status="completed"
)
print(f"保存导航记录成功，ID: {record.id}")
```

#### 获取导航记录

```python
# 获取用户的导航记录
user_records = db_service.get_user_navigation_records(user_id=1, limit=100)
```

## 性能优化建议

### 1. 速度优化

- **使用GPU加速**：确保在支持CUDA的环境中运行，YOLO会自动使用GPU
- **选择合适的模型**：根据设备性能选择不同大小的模型（n/s/m）
- **调整图像大小**：对于实时应用，使用较小的图像大小（如640x640）
- **批量处理**：如果需要处理大量图像，使用批量处理方式
- **缓存检测器**：使用单例模式缓存检测器实例，避免重复初始化

### 2. 准确度优化

- **调整置信度阈值**：根据场景需求调整置信度阈值
- **数据增强**：在训练时使用数据增强提高模型泛化能力
- **模型融合**：考虑使用多个模型的融合结果提高准确度

## 错误处理

### 常见错误及解决方案

1. **模型文件不存在**
   - 错误信息：`FileNotFoundError: [Errno 2] No such file or directory: 'models/best.pt'`
   - 解决方案：确保模型文件路径正确，或下载预训练模型

2. **ultralytics未安装**
   - 错误信息：`ImportError: ultralytics未安装，无法初始化YOLO检测器`
   - 解决方案：运行 `pip install ultralytics` 安装依赖

3. **数据库连接失败**
   - 错误信息：`sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file`
   - 解决方案：确保数据库文件路径正确，且具有写入权限

4. **图像读取失败**
   - 错误信息：`ValueError: 无法读取图像: path/to/image.jpg`
   - 解决方案：确保图像路径正确，且图像文件存在

## 测试

运行测试文件验证功能是否正常：

```bash
# 测试YOLO检测器
python -m tests.test_detector

# 测试数据库服务
python -m tests.test_database
```

## 示例代码

### 完整的前端集成示例

```python
from modules.detector import get_detector
from modules.database_service import get_database_service

# 初始化服务
detector = get_detector("models/best.pt")
db_service = get_database_service()

# 1. 用户登录
def login(username, password):
    # 实际应用中应该验证密码
    user = db_service.get_user_by_username(username)
    if user:
        return user
    return None

# 2. 图像检测
def detect_image(image_path, user_id):
    # 执行检测
    results = detector.detect_image(image_path)
    
    # 保存检测记录
    detection = db_service.save_detection(
        user_id=user_id,
        image_path=image_path,
        detection_result=results,
        detection_type="blind_road"
    )
    
    return results, detection.id

# 3. 获取检测历史
def get_detection_history(user_id, limit=50):
    return db_service.get_user_detections(user_id, limit=limit)

# 4. 获取检测统计
def get_detection_stats(user_id):
    return db_service.get_detection_statistics(user_id)

# 使用示例
if __name__ == "__main__":
    # 登录
    user = login("admin", "admin123")
    if user:
        print(f"登录成功: {user.username}")
        
        # 检测图像
        results, detection_id = detect_image("data/images/image_001.jpg", user.id)
        print(f"检测完成，记录ID: {detection_id}")
        print(f"检测到 {len(results)} 个目标")
        
        # 获取历史记录
        history = get_detection_history(user.id)
        print(f"历史检测记录: {len(history)} 条")
        
        # 获取统计信息
        stats = get_detection_stats(user.id)
        print(f"总检测次数: {stats['total_detections']}")
    else:
        print("登录失败")
```

## 联系信息

如有任何问题或需要进一步的帮助，请联系项目维护者。

---

**版本**: 1.0.0
**最后更新**: 2026-03-16
