"""
性能优化工具

优化点:
1. 异步并行处理
2. 缓存机制
3. 批处理
4. 降采样
"""
import logging
import asyncio
import time
from functools import wraps
from typing import Callable, Any
import hashlib

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        # 结果缓存
        self.cache = {}
        self.cache_ttl = 0.5  # 缓存有效期(秒)
    
    @staticmethod
    def async_parallel(tasks: list) -> list:
        """
        异步并行执行任务
        
        示例:
        results = await PerformanceOptimizer.async_parallel([
            vision_detect(),
            audio_classify(),
            trajectory_predict()
        ])
        """
        async def run_all():
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        return asyncio.run(run_all())
    
    def cached_result(self, ttl: float = None):
        """
        结果缓存装饰器
        
        用法:
        @optimizer.cached_result(ttl=1.0)
        async def expensive_function(input_data):
            ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # 检查缓存
                if cache_key in self.cache:
                    cached_data, cached_time = self.cache[cache_key]
                    age = time.time() - cached_time
                    
                    if age < (ttl or self.cache_ttl):
                        logger.debug(f"缓存命中: {func.__name__}")
                        return cached_data
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 保存缓存
                self.cache[cache_key] = (result, time.time())
                
                return result
            
            return wrapper
        return decorator
    
    @staticmethod
    def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_str = f"{func_name}_{args}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def downsample_image(image, target_size=(640, 480)):
        """
        图像降采样（减少计算量）
        
        Args:
            image: 原始图像
            target_size: 目标尺寸
        Returns:
            降采样后的图像
        """
        import cv2
        h, w = image.shape[:2]
        
        if w > target_size[0] or h > target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            logger.debug(f"图像降采样: {w}x{h} → {target_size[0]}x{target_size[1]}")
        
        return image
    
    @staticmethod
    def frame_skip(frame_count: int, skip_interval: int = 2) -> bool:
        """
        帧跳过策略（降低处理频率）
        
        Args:
            frame_count: 当前帧数
            skip_interval: 跳过间隔（处理1帧，跳过N帧）
        Returns:
            是否处理该帧
        """
        return frame_count % (skip_interval + 1) == 0


# 全局优化器
optimizer = PerformanceOptimizer()
