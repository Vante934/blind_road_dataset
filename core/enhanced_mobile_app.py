# -*- coding: utf-8 -*-
"""
盲道障碍检测移动端应用 - 增强版
集成YOLOv8检测 + 轨迹预测 + 智能语音预警
支持盲道识别、动态障碍物跟踪和轨迹预测
"""

import sys
import os
import cv2
import numpy as np
import json
import threading
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import requests
import base64
import tempfile
import uuid
from collections import deque
import math

# 导入轨迹预测模块
try:
    from trajectory_predictor import TrajectoryPredictor
    TRAJECTORY_PREDICTOR_AVAILABLE = True
    print("✅ 轨迹预测模块导入成功")
except ImportError:
    TRAJECTORY_PREDICTOR_AVAILABLE = False
    print("⚠️ 轨迹预测模块导入失败")

# 导入语音库
try:
    from voice_library import VoiceLibrary
    VOICE_LIBRARY_AVAILABLE = True
    print("✅ 语音库导入成功")
except ImportError:
    VOICE_LIBRARY_AVAILABLE = False
    print("⚠️ 语音库导入失败，将使用默认语音提示")

# 导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO导入成功")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO导入失败")

class BaiduTTS:
    """百度语音合成"""
    
    def __init__(self, app_id, api_key, secret_key):
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_expire_time = 0
        
    def get_access_token(self):
        """获取访问令牌"""
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token
            
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        try:
            response = requests.post(url, params=params)
            result = response.json()
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                self.token_expire_time = time.time() + result["expires_in"] - 60
                print("✅ 百度语音令牌获取成功")
                return self.access_token
            else:
                print(f"❌ 获取百度语音令牌失败: {result}")
                return None
        except Exception as e:
            print(f"❌ 获取百度语音令牌异常: {e}")
            return None
    
    def text_to_speech(self, text, output_file=None):
        """文本转语音"""
        token = self.get_access_token()
        if not token:
            return False
            
        url = "https://tsn.baidu.com/text2audio"
        
        data = {
            "tex": text,
            "tok": token,
            "cuid": "blind_road_detector",
            "ctp": "1",
            "lan": "zh",
            "spd": "5",
            "pit": "5",
            "vol": "5",
            "per": "0"
        }
        
        try:
            response = requests.post(url, data=data)
            
            if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('audio'):
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ 语音文件保存: {output_file}")
                return True
            else:
                print(f"❌ 语音合成失败: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 语音合成异常: {e}")
            return False

class SimpleVoiceSystem:
    """简化语音系统 - 优化版本，支持连续完整播报"""
    
    def __init__(self):
        self.media_player = None
        self.baidu_tts = None
        self.voice_enabled = True
        self.last_speech_time = 0
        self.speech_cooldown = 1.0  # 减少冷却时间
        
        # 语音队列管理
        self.voice_queue = []
        self.is_playing = False
        self.voice_lock = threading.Lock()
        
        self.init_media_player()
        self.init_baidu_tts()
        
        # 启动语音处理线程
        self.start_voice_processor()
    
    def init_media_player(self):
        """初始化媒体播放器"""
        try:
            self.media_player = QMediaPlayer()
            print("✅ 媒体播放器初始化成功")
        except Exception as e:
            print(f"❌ 媒体播放器初始化失败: {e}")
    
    def init_baidu_tts(self):
        """初始化百度TTS"""
        try:
            # 从配置文件读取百度语音配置
            if os.path.exists("voice_config.json"):
                with open("voice_config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                baidu_config = config.get('baidu_tts', {})
                app_id = baidu_config.get('app_id')
                api_key = baidu_config.get('api_key')
                secret_key = baidu_config.get('secret_key')
                
                if app_id and api_key and secret_key:
                    self.baidu_tts = BaiduTTS(app_id, api_key, secret_key)
                    print("✅ 百度TTS初始化成功")
                else:
                    print("⚠️ 百度TTS配置不完整")
            else:
                print("⚠️ 语音配置文件不存在")
        except Exception as e:
            print(f"❌ 百度TTS初始化失败: {e}")
    
    def start_voice_processor(self):
        """启动语音处理线程"""
        def voice_processor():
            while True:
                try:
                    with self.voice_lock:
                        if self.voice_queue and not self.is_playing:
                            text = self.voice_queue.pop(0)
                            self.is_playing = True
                        else:
                            text = None
                    
                    if text:
                        self.execute_speech(text)
                        with self.voice_lock:
                            self.is_playing = False
                    else:
                        time.sleep(0.1)  # 短暂等待
                        
                except Exception as e:
                    print(f"⚠️ 语音处理线程错误: {e}")
                    with self.voice_lock:
                        self.is_playing = False
        
        voice_thread = threading.Thread(target=voice_processor, daemon=True)
        voice_thread.start()
        print("✅ 语音处理线程已启动")
    
    def speak(self, text, priority=1):
        """播放语音 - 支持优先级和队列管理"""
        if not self.voice_enabled:
            return
        
        # 检查冷却时间（根据优先级调整）
        current_time = time.time()
        cooldown = 0.5 if priority >= 3 else 1.0
        if current_time - self.last_speech_time < cooldown:
            return
        
        self.last_speech_time = current_time
        
        print(f"🔊 语音播报 (优先级{priority}): {text}")
        
        # 添加到语音队列
        with self.voice_lock:
            if priority >= 3:
                # 高优先级，插入到队列前面
                self.voice_queue.insert(0, text)
            else:
                # 普通优先级，添加到队列末尾
                self.voice_queue.append(text)
    
    def execute_speech(self, text):
        """执行语音播报"""
        try:
            if self.baidu_tts:
                # 使用百度TTS
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                temp_file.close()
                
                if self.baidu_tts.text_to_speech(text, temp_file.name):
                    self.play_audio(temp_file.name)
                    # 延迟删除临时文件
                    threading.Timer(5.0, lambda: os.unlink(temp_file.name) if os.path.exists(temp_file.name) else None).start()
                else:
                    print(f"⚠️ 使用默认语音提示: {text}")
            else:
                # 备用方案：使用pyttsx3
                try:
                    import pyttsx3
                    tts = pyttsx3.init()
                    tts.setProperty('rate', 150)
                    tts.say(text)
                    tts.runAndWait()
                    tts.stop()
                    print(f"✅ 备用语音播报成功: {text}")
                except Exception as e:
                    print(f"⚠️ 备用语音播报失败: {e}")
        except Exception as e:
            print(f"❌ 语音播放失败: {e}")
    
    def play_audio(self, audio_file):
        """播放音频文件"""
        try:
            if self.media_player:
                url = QUrl.fromLocalFile(audio_file)
                content = QMediaContent(url)
                self.media_player.setMedia(content)
                self.media_player.play()
                print(f"✅ 音频播放成功: {audio_file}")
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

class EnhancedMobileDetectWindow(QMainWindow):
    """增强版移动端检测窗口"""
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.timer = QTimer()
        self.voice_system = SimpleVoiceSystem()
        self.trajectory_predictor = None
        self.yolo_model = None
        self.detection_history = deque(maxlen=100)
        
        # 初始化模型
        self.init_models()
        self.init_ui()
        
        # 连接信号
        self.timer.timeout.connect(self.update_frame)
    
    def init_models(self):
        """初始化检测和跟踪模型"""
        print("🔧 初始化模型...")
        
        # 初始化YOLO检测模型
        if YOLO_AVAILABLE:
            try:
                # 优先使用自定义模型
                if os.path.exists("runs/detect/train5/weights/best.pt"):
                    self.yolo_model = YOLO("runs/detect/train5/weights/best.pt")
                    print("✅ 加载自定义YOLO模型")
                elif os.path.exists("yolov8n.pt"):
                    self.yolo_model = YOLO("yolov8n.pt")
                    print("✅ 加载默认YOLO模型")
                else:
                    print("❌ 未找到YOLO模型文件")
                    self.yolo_model = None
            except Exception as e:
                print(f"❌ YOLO模型加载失败: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None
        
        # 初始化轨迹预测器
        if TRAJECTORY_PREDICTOR_AVAILABLE:
            self.trajectory_predictor = TrajectoryPredictor()
            print("✅ 轨迹预测器初始化成功")
        else:
            self.trajectory_predictor = None
            print("⚠️ 轨迹预测器不可用")
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("盲道障碍检测 - 增强版 (轨迹预测)")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧视频和控制区域
        left_panel = QVBoxLayout()
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("点击开始检测")
        left_panel.addWidget(self.video_label)
        
        # 控制按钮区域
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.start_button)
        
        self.voice_button = QPushButton("语音开关")
        self.voice_button.setCheckable(True)
        self.voice_button.setChecked(True)
        self.voice_button.clicked.connect(self.toggle_voice)
        button_layout.addWidget(self.voice_button)
        
        self.test_voice_button = QPushButton("测试语音")
        self.test_voice_button.clicked.connect(self.test_voice)
        button_layout.addWidget(self.test_voice_button)
        
        left_panel.addLayout(button_layout)
        
        # 状态显示区域
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        left_panel.addWidget(self.status_label)
        
        main_layout.addLayout(left_panel)
        
        # 右侧信息显示区域
        right_panel = QVBoxLayout()
        
        # 跟踪信息显示
        self.tracking_label = QLabel("跟踪目标: 0")
        self.tracking_label.setStyleSheet("color: blue; font-weight: bold;")
        right_panel.addWidget(self.tracking_label)
        
        # 盲道信息显示
        self.blind_path_label = QLabel("盲道状态: 未检测")
        self.blind_path_label.setStyleSheet("color: orange; font-weight: bold;")
        right_panel.addWidget(self.blind_path_label)
        
        # 碰撞风险显示
        self.risk_label = QLabel("碰撞风险: 低")
        self.risk_label.setStyleSheet("color: green; font-weight: bold;")
        right_panel.addWidget(self.risk_label)
        
        # 警告信息显示区域
        self.warning_text = QTextEdit()
        self.warning_text.setMaximumHeight(150)
        self.warning_text.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.warning_text.setReadOnly(True)
        right_panel.addWidget(QLabel("警告信息:"))
        right_panel.addWidget(self.warning_text)
        
        # 环境安全信息显示
        self.environment_label = QLabel("环境安全: 良好")
        self.environment_label.setStyleSheet("color: green; font-weight: bold;")
        right_panel.addWidget(self.environment_label)
        
        # 安全评分显示
        self.safety_score_label = QLabel("安全评分: 100%")
        self.safety_score_label.setStyleSheet("color: green; font-weight: bold;")
        right_panel.addWidget(self.safety_score_label)
        
        # 天气信息显示
        self.weather_label = QLabel("天气条件: 晴朗")
        self.weather_label.setStyleSheet("color: blue; font-weight: bold;")
        right_panel.addWidget(self.weather_label)
        
        # 光照信息显示
        self.lighting_label = QLabel("光照条件: 正常")
        self.lighting_label.setStyleSheet("color: blue; font-weight: bold;")
        right_panel.addWidget(self.lighting_label)
        
        # 路面信息显示
        self.surface_label = QLabel("路面条件: 平整")
        self.surface_label.setStyleSheet("color: blue; font-weight: bold;")
        right_panel.addWidget(self.surface_label)
        
        # 安全指导显示
        self.guidance_label = QLabel("安全指导: 环境安全，可以正常前进")
        self.guidance_label.setStyleSheet("color: green; font-weight: bold;")
        self.guidance_label.setWordWrap(True)
        right_panel.addWidget(self.guidance_label)
        
        main_layout.addLayout(right_panel)
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if self.camera is None or not self.camera.isOpened():
            self.start_camera()
        else:
            self.stop_camera()
    
    def toggle_voice(self):
        """切换语音开关"""
        self.voice_system.voice_enabled = self.voice_button.isChecked()
        status = "开启" if self.voice_system.voice_enabled else "关闭"
        self.status_label.setText(f"状态: 语音{status}")
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.start_button.setText("停止检测")
                self.status_label.setText("状态: 检测中")
                self.timer.start(30)  # 30ms间隔，约33FPS
                print("✅ 摄像头启动成功")
            else:
                print("❌ 摄像头启动失败")
                self.status_label.setText("状态: 摄像头启动失败")
        except Exception as e:
            print(f"❌ 摄像头启动异常: {e}")
            self.status_label.setText("状态: 摄像头启动异常")
    
    def stop_camera(self):
        """停止摄像头"""
        self.timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_button.setText("开始检测")
        self.status_label.setText("状态: 已停止")
        self.video_label.setText("点击开始检测")
        print("✅ 摄像头已停止")
    
    def update_frame(self):
        """更新视频帧"""
        if not self.camera or not self.camera.isOpened():
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # 调整帧大小
        frame = cv2.resize(frame, (640, 480))
        
        # 执行检测和轨迹预测
        processed_frame = self.process_frame(frame)
        
        # 转换为Qt格式并显示
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
    
    def process_frame(self, frame):
        """处理视频帧：检测 + 轨迹预测 + 预警"""
        if not self.yolo_model or not self.trajectory_predictor:
            return frame
        
        try:
            # Step 1: YOLOv8检测
            results = self.yolo_model(frame, stream=True)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, clss):
                        if conf > 0.5:  # 置信度阈值
                            x1, y1, x2, y2 = box
                            detections.append([x1, y1, x2, y2, conf, int(cls)])
            
            # Step 2: 轨迹预测处理
            prediction_result = self.trajectory_predictor.process_frame(frame, detections)
            
            # Step 3: 绘制检测结果和轨迹
            frame = self.draw_detections_and_trajectories(frame, detections, prediction_result)
            
            # Step 4: 更新UI信息
            self.update_ui_info(prediction_result)
            
            # Step 5: 处理警告和语音提示
            self.handle_warnings(prediction_result)
            
        except Exception as e:
            print(f"❌ 帧处理异常: {e}")
        
        return frame
    
    def draw_detections_and_trajectories(self, frame, detections, prediction_result):
        """绘制检测结果和轨迹"""
        # 绘制检测框
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            color = self.get_class_color(cls)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 添加标签
            label = f"{self.get_class_name(cls)} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制盲道信息
        if prediction_result.get('blind_path'):
            blind_path = prediction_result['blind_path']
            center = blind_path['center']
            width = blind_path['width']
            height = blind_path['height']
            confidence = blind_path['confidence']
            
            # 绘制盲道轮廓
            if 'contour' in blind_path:
                cv2.drawContours(frame, [blind_path['contour']], -1, (0, 255, 255), 2)
            
            # 绘制盲道中心线
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            cv2.putText(frame, f"盲道 {confidence:.2f}", (center[0]-50, center[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 绘制预测轨迹
            if 'predicted_trajectory' in blind_path:
                predicted_points = blind_path['predicted_trajectory']
                for i, point in enumerate(predicted_points):
                    color = (255, 255, 0) if i == len(predicted_points) - 1 else (100, 100, 100)
                    cv2.circle(frame, point, 3, color, -1)
        
        # 绘制跟踪目标和轨迹
        tracked_objects = prediction_result.get('tracked_objects', [])
        for obj in tracked_objects:
            track_id = obj['id']
            centroid = obj['centroid']
            class_id = obj.get('class_id', 0)
            
            # 绘制跟踪框
            x1, y1 = centroid[0] - 25, centroid[1] - 25
            x2, y2 = centroid[0] + 25, centroid[1] + 25
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制历史轨迹
            if 'trajectory' in obj and len(obj['trajectory']) > 1:
                trajectory = obj['trajectory']
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 2)
            
            # 绘制预测轨迹
            if 'predicted_trajectory' in obj:
                predicted_points = obj['predicted_trajectory']
                for i, point in enumerate(predicted_points):
                    color = (255, 0, 0) if i == len(predicted_points) - 1 else (100, 100, 100)
                    cv2.circle(frame, point, 3, color, -1)
        
        # 绘制用户位置（帧中心）
        user_pos = (320, 240)
        cv2.circle(frame, user_pos, 10, (255, 255, 255), -1)
        cv2.putText(frame, "用户", (user_pos[0]-20, user_pos[1]+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def update_ui_info(self, prediction_result):
        """更新UI信息"""
        # 更新跟踪目标数量
        tracked_objects = prediction_result.get('tracked_objects', [])
        self.tracking_label.setText(f"跟踪目标: {len(tracked_objects)}")
        
        # 更新盲道状态
        blind_path = prediction_result.get('blind_path')
        if blind_path:
            confidence = blind_path.get('confidence', 0)
            self.blind_path_label.setText(f"盲道状态: 已检测 (置信度: {confidence:.2f})")
            self.blind_path_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.blind_path_label.setText("盲道状态: 未检测")
            self.blind_path_label.setStyleSheet("color: red; font-weight: bold;")
        
        # 更新碰撞风险
        collision_risks = prediction_result.get('collision_risks', {})
        if collision_risks:
            max_risk = max(collision_risks.values()) if collision_risks else 0
            if max_risk > 0.7:
                self.risk_label.setText(f"碰撞风险: 高 ({max_risk:.2f})")
                self.risk_label.setStyleSheet("color: red; font-weight: bold;")
            elif max_risk > 0.3:
                self.risk_label.setText(f"碰撞风险: 中 ({max_risk:.2f})")
                self.risk_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.risk_label.setText(f"碰撞风险: 低 ({max_risk:.2f})")
                self.risk_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.risk_label.setText("碰撞风险: 低")
            self.risk_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 更新环境安全信息
        self.update_environment_info(prediction_result)
        
        # 更新安全指导
        if self.trajectory_predictor:
            guidance = self.trajectory_predictor.get_safety_guidance()
            self.guidance_label.setText(f"安全指导: {guidance}")
            
            # 根据指导内容设置颜色
            if "紧急" in guidance or "危险" in guidance:
                self.guidance_label.setStyleSheet("color: red; font-weight: bold;")
            elif "注意" in guidance:
                self.guidance_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.guidance_label.setStyleSheet("color: green; font-weight: bold;")
    
    def update_environment_info(self, prediction_result):
        """更新环境安全信息显示"""
        # 更新环境安全等级
        safety_level = prediction_result.get('overall_safety_level', 'safe')
        safety_score = prediction_result.get('safety_score', 1.0)
        
        if safety_level == 'high_risk':
            self.environment_label.setText("环境安全: 高风险")
            self.environment_label.setStyleSheet("color: red; font-weight: bold;")
        elif safety_level == 'medium_risk':
            self.environment_label.setText("环境安全: 中等风险")
            self.environment_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.environment_label.setText("环境安全: 良好")
            self.environment_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 更新安全评分
        score_percentage = int(safety_score * 100)
        self.safety_score_label.setText(f"安全评分: {score_percentage}%")
        
        if score_percentage < 30:
            self.safety_score_label.setStyleSheet("color: red; font-weight: bold;")
        elif score_percentage < 60:
            self.safety_score_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.safety_score_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 更新天气信息
        weather_info = prediction_result.get('weather_info')
        if weather_info:
            weather_type = weather_info.get('weather_type', 'clear')
            visibility = weather_info.get('visibility_level', 'good')
            self.weather_label.setText(f"天气条件: {weather_type} ({visibility})")
            
            if weather_info.get('safety_impact') == 'very_high':
                self.weather_label.setStyleSheet("color: red; font-weight: bold;")
            elif weather_info.get('safety_impact') == 'high':
                self.weather_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.weather_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.weather_label.setText("天气条件: 晴朗")
            self.weather_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # 更新光照信息
        lighting_info = prediction_result.get('lighting_info')
        if lighting_info:
            lighting_level = lighting_info.get('lighting_level', 'normal')
            visibility_quality = lighting_info.get('visibility_quality', 'good')
            self.lighting_label.setText(f"光照条件: {lighting_level} ({visibility_quality})")
            
            if lighting_info.get('safety_impact') == 'very_high':
                self.lighting_label.setStyleSheet("color: red; font-weight: bold;")
            elif lighting_info.get('safety_impact') == 'high':
                self.lighting_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.lighting_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.lighting_label.setText("光照条件: 正常")
            self.lighting_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # 更新路面信息
        surface_info = prediction_result.get('surface_info')
        if surface_info:
            surface_type = surface_info.get('surface_type', 'smooth')
            safety_level = surface_info.get('safety_level', 'safe')
            self.surface_label.setText(f"路面条件: {surface_type} ({safety_level})")
            
            if surface_info.get('safety_level') == 'caution':
                self.surface_label.setStyleSheet("color: orange; font-weight: bold;")
            elif surface_info.get('safety_level') == 'moderate':
                self.surface_label.setStyleSheet("color: blue; font-weight: bold;")
            else:
                self.surface_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.surface_label.setText("路面条件: 平整")
            self.surface_label.setStyleSheet("color: green; font-weight: bold;")
    
    def handle_warnings(self, prediction_result):
        """处理警告和语音提示"""
        warnings = prediction_result.get('warnings', [])
        emergency_alerts = prediction_result.get('emergency_alerts', [])
        
        # 更新警告文本
        all_warnings = warnings + emergency_alerts
        if all_warnings:
            warning_text = "\n".join(all_warnings)
            self.warning_text.append(f"[{time.strftime('%H:%M:%S')}] {warning_text}")
            
            # 语音播报第一个警告（优先紧急警报）
            if all_warnings and self.voice_system.voice_enabled:
                priority_warning = emergency_alerts[0] if emergency_alerts else warnings[0]
                self.voice_system.speak(priority_warning, priority=3 if emergency_alerts else 1)
        
        # 滚动到底部
        self.warning_text.verticalScrollBar().setValue(
            self.warning_text.verticalScrollBar().maximum()
        )
    
    def get_class_color(self, cls):
        """获取类别颜色"""
        colors = [
            (255, 0, 0),    # 红色 - 人
            (0, 255, 0),    # 绿色 - 车
            (0, 0, 255),    # 蓝色 - 其他
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
        ]
        return colors[cls % len(colors)]
    
    def get_class_name(self, cls):
        """获取类别名称"""
        names = ["人", "车", "障碍物", "坑洼", "其他"]
        return names[cls % len(names)]
    
    def test_voice(self):
        """测试语音"""
        self.voice_system.speak("轨迹预测系统测试正常")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_camera()
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = EnhancedMobileDetectWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 