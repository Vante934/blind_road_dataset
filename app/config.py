from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ========== 原有配置（✅ 完全不变）==========
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    
    DATABASE_URL: str = ""
    
    BAIDU_APP_ID: str = ""
    BAIDU_API_KEY: str = ""
    BAIDU_SECRET_KEY: str = ""
    AUDIO_ENGINE: str = "baidu"
    
    DISTANCE_LEVEL1: float = 0.5
    DISTANCE_LEVEL2: float = 1.5
    DISTANCE_LEVEL3: float = 3.0
    
    # ========== 🆕 模块开关（默认全关闭，保证兼容）==========
    # 视觉检测模块
    MODULE_VISION_ENABLED: bool = True
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_DEVICE: str = "cpu"
    YOLO_CONFIDENCE: float = 0.5
    
    # 轨迹分析模块
    MODULE_TRAJECTORY_ENABLED: bool = True
    TRAJECTORY_WINDOW_SIZE: int = 20
    
    # 环境检测模块（未来）
    MODULE_ENVIRONMENT_ENABLED: bool = False
    
    # 路径规划
    MODULE_ROUTE_PLANNING_ENABLED: bool = True
    
    # 深度估计
    CAMERA_FOCAL_LENGTH: float = 1000.0
    CAMERA_HEIGHT: float = 1.5
    ENABLE_MONOCULAR_DEPTH: bool = False
    
    # 性能优化
    FRAME_SKIP_INTERVAL: int = 0
    IMAGE_DOWNSAMPLE: bool = False
    IMAGE_TARGET_SIZE: tuple = (640, 480)
    
    class Config:
        env_file = ".env"


settings = Settings()