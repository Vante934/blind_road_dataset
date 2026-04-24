"""
完整系统测试

测试场景:
1. 静态障碍物检测
2. 动态障碍物接近
3. 声音预警
4. 多威胁融合
5. 路径规划
"""
import asyncio
import pytest
import cv2
import numpy as np
import base64
from app.services.integrated_pipeline import integrated_pipeline
from app.models.schemas import SensorData, AudioData


class TestIntegratedSystem:
    """完整系统测试"""
    
    @pytest.mark.asyncio
    async def test_static_obstacle_detection(self):
        """测试场景1: 静态障碍物检测"""
        print("\n===== 场景1: 静态障碍物（电线杆）=====")
        
        # 模拟图像（实际应使用真实图片）
        image = self._create_mock_image_with_pole()
        img_b64 = self._image_to_base64(image)
        
        # 构建传感器数据
        sensor_data = SensorData(
            device_id="test_001",
            timestamp=1700000000000,
            tof_distance=2.0,
            tof_direction="center",
            video_frame=img_b64
        )
        
        # 处理
        result = await integrated_pipeline.process(sensor_data)
        
        # 断言
        assert result.success
        assert len(result.vision_obstacles) > 0
        
        # 检查预警
        assert result.warning_decision is not None
        warning_level = result.warning_decision["warning_level"]
        
        print(f"检测到 {len(result.vision_obstacles)} 个障碍物")
        print(f"预警级别: {warning_level}")
        print(f"TTS: {result.warning_decision['tts_text']}")
        
        # 静态障碍物2米外，应该是2级或3级
        assert warning_level in [2, 3]
    
    @pytest.mark.asyncio
    async def test_dynamic_approaching(self):
        """测试场景2: 动态障碍物快速接近"""
        print("\n===== 场景2: 汽车快速接近 =====")
        
        # 模拟连续帧（距离递减）
        distances = [5.0, 4.0, 3.0, 2.0, 1.0]
        
        for i, dist in enumerate(distances):
            image = self._create_mock_image_with_car(dist)
            img_b64 = self._image_to_base64(image)
            
            sensor_data = SensorData(
                device_id="test_002",
                timestamp=1700000000000 + i * 500,  # 每0.5秒一帧
                tof_distance=dist,
                tof_direction="center",
                video_frame=img_b64
            )
            
            result = await integrated_pipeline.process(sensor_data)
            
            if result.warning_decision:
                level = result.warning_decision["warning_level"]
                tts = result.warning_decision["tts_text"]
                print(f"距离{dist}m → 预警级别{level}: {tts}")
            
            await asyncio.sleep(0.1)
        
        # 最后一次应该是一级预警
        assert result.warning_decision["warning_level"] == 1
    
    @pytest.mark.asyncio
    async def test_sound_warning(self):
        """测试场景3: 声音预警（鸣笛）"""
        print("\n===== 场景3: 汽车鸣笛声 =====")
        
        # 模拟鸣笛声音频（这里用假数据）
        audio_data = self._create_mock_horn_audio()
        audio_b64 = base64.b64encode(audio_data).decode()
        
        sensor_data = SensorData(
            device_id="test_003",
            timestamp=1700000000000,
            tof_distance=3.0,
            audio_data=AudioData(
                audio_base64=audio_b64,
                audio_format="pcm",
                sample_rate=16000
            )
        )
        
        result = await integrated_pipeline.process(sensor_data)
        
        # 检查声音识别
        assert result.sound_result is not None
        print(f"识别到: {result.sound_result['sound_label']}")
        
        # 有鸣笛声，应该提升预警级别
        assert result.warning_decision["warning_level"] >= 2
    
    @pytest.mark.asyncio
    async def test_multi_threat_fusion(self):
        """测试场景4: 多威胁融合"""
        print("\n===== 场景4: 近距离 + 动态障碍 + 鸣笛 =====")
        
        # 同时存在多个威胁
        image = self._create_mock_image_with_car(1.0)  # 1米
        img_b64 = self._image_to_base64(image)
        
        audio_data = self._create_mock_horn_audio()
        audio_b64 = base64.b64encode(audio_data).decode()
        
        sensor_data = SensorData(
            device_id="test_004",
            timestamp=1700000000000,
            tof_distance=1.0,
            tof_direction="center",
            video_frame=img_b64,
            audio_data=AudioData(
                audio_base64=audio_b64,
                audio_format="pcm",
                sample_rate=16000
            )
        )
        
        result = await integrated_pipeline.process(sensor_data)
        
        # 多威胁应该触发一级预警
        assert result.warning_decision["warning_level"] == 1
        
        # 检查威胁分解
        breakdown = result.warning_decision["threat_breakdown"]
        print(f"威胁分解: {breakdown}")
        
        # 应该有高分
        assert breakdown["total"] > 0.7
    
    @pytest.mark.asyncio
    async def test_route_planning(self):
        """测试场景5: 路径规划"""
        print("\n===== 场景5: 前方障碍，左右绕行 =====")
        
        # 前方有障碍
        image = self._create_mock_image_with_obstacle_center()
        img_b64 = self._image_to_base64(image)
        
        sensor_data = SensorData(
            device_id="test_005",
            timestamp=1700000000000,
            tof_distance=2.0,
            tof_direction="center",
            video_frame=img_b64
        )
        
        result = await integrated_pipeline.process(sensor_data)
        
        # 检查路径规划
        assert result.route_plan is not None
        
        recommended = result.route_plan["recommended"]
        print(f"推荐方向: {recommended['direction']}")
        print(f"原因: {recommended['reason']}")
        
        # 前方有障碍，应该建议绕行或停止
        assert recommended["direction"] in ["left", "right", "stop"]
    
    # ===== 辅助方法 =====
    
    def _create_mock_image_with_pole(self):
        """创建带电线杆的模拟图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # 绘制一个竖线（模拟电线杆）
        cv2.rectangle(img, (300, 100), (340, 400), (128, 128, 128), -1)
        return img
    
    def _create_mock_image_with_car(self, distance: float):
        """创建带汽车的模拟图像（距离越近，车越大）"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 距离 → bbox大小
        scale = 5.0 / distance
        w = int(200 * scale)
        h = int(150 * scale)
        
        x1 = 320 - w // 2
        y1 = 480 - h - 50
        x2 = x1 + w
        y2 = y1 + h
        
        # 绘制矩形（模拟车）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        return img
    
    def _create_mock_image_with_obstacle_center(self):
        """创建前方有障碍的图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (280, 200), (360, 400), (100, 100, 100), -1)
        return img
    
    def _image_to_base64(self, image):
        """图像转Base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode()
    
    def _create_mock_horn_audio(self):
        """创建模拟鸣笛音频（单频音）"""
        # 生成440Hz正弦波（模拟鸣笛）
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 16000
        audio = audio.astype(np.int16)
        
        return audio.tobytes()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
