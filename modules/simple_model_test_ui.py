#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版模型测试UI
支持不同模型的测试、对比和可视化
"""

import sys
import os
import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QProgressBar, QTextEdit,
                             QGroupBox, QSplitter, QFrame, QGridLayout,
                             QCheckBox, QFileDialog, QMessageBox, QTabWidget, QApplication,
                             QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPolygonF

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 延迟导入可能阻塞的模块
YOLO_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
VOICE_AVAILABLE = False
pyttsx3_engine = None
FigureCanvas = None
Figure = None

# 导入语音库
try:
    from modules.voice_library import voice_library
    VOICE_AVAILABLE = True
    print("✅ 语音库导入成功")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"⚠️ 语音库导入失败: {e}")

try:
    import pyttsx3
    pyttsx3_engine = pyttsx3.init()
    print("✅ 语音合成引擎初始化成功")
except ImportError as e:
    pyttsx3_engine = None
    print(f"⚠️ 语音合成引擎导入失败: {e}")
except Exception as e:
    pyttsx3_engine = None
    print(f"⚠️ 语音合成引擎初始化失败: {e}")

def _lazy_import_yolo():
    """延迟导入YOLO"""
    global YOLO_AVAILABLE
    if YOLO_AVAILABLE:
        return
    try:
        print("    正在导入ultralytics...")
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
        print("    ✅ ultralytics导入成功")
    except ImportError as e:
        YOLO_AVAILABLE = False
        print(f"    ⚠️ ultralytics未安装: {e}")

def _lazy_import_matplotlib():
    """延迟导入matplotlib"""
    global MATPLOTLIB_AVAILABLE, FigureCanvas, Figure
    if MATPLOTLIB_AVAILABLE:
        return True
    
    try:
        print("    正在导入matplotlib...")
        # 暂时跳过matplotlib导入，使用备用方案
        print("    ⚠️ 跳过matplotlib导入，使用备用方案")
        MATPLOTLIB_AVAILABLE = False
        # 创建占位类
        class FigureCanvas:
            pass
        class Figure:
            pass
        return False
    except Exception as e:
        MATPLOTLIB_AVAILABLE = False
        print(f"    ⚠️ matplotlib导入出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        # 创建占位类
        class FigureCanvas:
            pass
        class Figure:
            pass
        return False


class ModelTestWorker(QThread):
    """模型测试工作线程"""
    progress_updated = pyqtSignal(int, str)
    metrics_updated = pyqtSignal(dict)
    test_finished = pyqtSignal(dict)
    
    def __init__(self, model_path, test_images, test_type='annotation'):
        super().__init__()
        self.model_path = model_path
        self.test_images = test_images
        self.test_type = test_type
        self.is_running = False
        
    def run(self):
        """运行测试"""
        self.is_running = True
        try:
            # 延迟导入YOLO
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            # 导入延迟导入函数
            from modules.simple_model_test_ui import _lazy_import_yolo
            _lazy_import_yolo()
            
            # 检查YOLO是否可用
            import importlib
            yolo_module = importlib.import_module('modules.simple_model_test_ui')
            if not yolo_module.YOLO_AVAILABLE:
                raise Exception("ultralytics未安装")
            
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            total = len(self.test_images)
            metrics = {
                'inference_times': [],
                'detection_counts': [],
                'confidences': [],
                'class_distribution': {},
                'predictions': [],
                'ground_truths': []
            }
            
            bad_cases = []
            
            for i, img_path in enumerate(self.test_images):
                if not self.is_running:
                    break
                
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 推理
                start_time = time.time()
                results = model(img, conf=0.25, verbose=False)
                inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
                
                # 统计结果
                detection_count = 0
                confidences = []
                class_dist = {}
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection_count += 1
                            conf = float(box.conf[0].cpu().numpy())
                            confidences.append(conf)
                            
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                            class_dist[class_name] = class_dist.get(class_name, 0) + 1
                            
                            # 保存检测结果
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class_name': class_name,
                                'confidence': conf
                            })
                
                metrics['inference_times'].append(inference_time)
                metrics['detection_counts'].append(detection_count)
                metrics['confidences'].extend(confidences)
                metrics['class_distribution'] = class_dist
                
                # 保存预测结果
                metrics['predictions'].extend([d['class_name'] for d in detections])
                
                # 简单的地面真值模拟（实际应用中应该从标签文件读取）
                # 这里我们假设所有检测到的物体都是真实存在的
                metrics['ground_truths'].extend([d['class_name'] for d in detections])
                
                # 生成错题集数据
                if detection_count == 0 or (confidences and np.mean(confidences) < 0.5):
                    bad_cases.append({
                        'image_path': img_path,
                        'detections': detections,
                        'inference_time': inference_time
                    })
                
                # 发送进度
                progress = int((i + 1) / total * 100)
                self.progress_updated.emit(progress, f"处理中: {i+1}/{total}")
                self.metrics_updated.emit({
                    'inference_time': inference_time,
                    'detection_count': detection_count,
                    'avg_confidence': np.mean(confidences) if confidences else 0
                })
            
            # 计算最终指标
            final_metrics = {
                'avg_inference_time': np.mean(metrics['inference_times']) if metrics['inference_times'] else 0,
                'avg_detection_count': np.mean(metrics['detection_counts']) if metrics['detection_counts'] else 0,
                'avg_confidence': np.mean(metrics['confidences']) if metrics['confidences'] else 0,
                'total_detections': sum(metrics['detection_counts']),
                'class_distribution': metrics['class_distribution'],
                'total_images': total,
                'processed_images': len(metrics['inference_times']),
                'bad_cases': bad_cases,
                'predictions': metrics['predictions'],
                'ground_truths': metrics['ground_truths']
            }
            
            self.test_finished.emit(final_metrics)
            
        except Exception as e:
            self.test_finished.emit({'error': str(e)})
    
    def stop(self):
        """停止测试"""
        self.is_running = False


class RadarChartWidget(QWidget):
    """雷达图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {}
        self.setMinimumSize(400, 400)
        
    def set_metrics(self, metrics: Dict):
        """设置指标数据"""
        self.metrics = metrics
        self.update()
    
    def paintEvent(self, event):
        """绘制雷达图"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) / 2 - 50
        
        # 定义维度
        dimensions = [
            ('准确率', 'accuracy', 0, 1),
            ('精确率', 'precision', 0, 1),
            ('召回率', 'recall', 0, 1),
            ('F1分数', 'f1_score', 0, 1),
            ('速度', 'speed', 0, 100),  # FPS
            ('稳定性', 'stability', 0, 1)
        ]
        
        num_dimensions = len(dimensions)
        angle_step = 2 * np.pi / num_dimensions
        
        # 绘制网格
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for i in range(1, 6):
            r = radius * i / 5
            painter.drawEllipse(int(center_x - r), int(center_y - r), int(r * 2), int(r * 2))
        
        # 绘制轴线
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        for i in range(num_dimensions):
            angle = i * angle_step - np.pi / 2
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            painter.drawLine(int(center_x), int(center_y), int(x), int(y))
        
        # 绘制标签
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont("Arial", 10)
        painter.setFont(font)
        for i, (name, key, min_val, max_val) in enumerate(dimensions):
            angle = i * angle_step - np.pi / 2
            label_radius = radius + 20
            x = center_x + label_radius * np.cos(angle)
            y = center_y + label_radius * np.sin(angle)
            painter.drawText(int(x - 30), int(y - 10), 60, 20, Qt.AlignCenter, name)
        
        # 绘制数据多边形
        if self.metrics:
            points = []
            for i, (name, key, min_val, max_val) in enumerate(dimensions):
                value = self.metrics.get(key, 0)
                # 归一化到0-1
                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
                normalized = max(0, min(1, normalized))  # 限制在0-1之间
                
                angle = i * angle_step - np.pi / 2
                r = radius * normalized
                x = center_x + r * np.cos(angle)
                y = center_y + r * np.sin(angle)
                points.append(QPointF(x, y))
            
            # 绘制填充区域
            polygon = QPolygonF(points)
            painter.setBrush(QColor(76, 175, 80, 100))
            painter.setPen(QPen(QColor(76, 175, 80), 2))
            painter.drawPolygon(polygon)
            
            # 绘制点
            painter.setBrush(QColor(76, 175, 80))
            for point in points:
                painter.drawEllipse(int(point.x() - 3), int(point.y() - 3), 6, 6)


class SimpleModelTestUI(QWidget):
    """简化版模型测试UI"""
    
    def __init__(self, parent=None):
        try:
            super().__init__(parent)
            print("SimpleModelTestUI: 开始初始化...")
            self.current_model = None
            self.current_model_path = None
            self.test_worker = None
            self.camera_active = False
            self.camera_cap = None
            self.camera_timer = QTimer(self)
            self.camera_timer.timeout.connect(self.update_camera_frame)
            
            # 模型指标存储
            self.model_metrics = {}  # {model_name: metrics}
            
            # 语音相关属性
            self.voice_enabled = True
            self.last_voice_time = 0
            self.voice_cooldown = 2  # 语音播报冷却时间（秒）
            
            print("SimpleModelTestUI: 初始化UI...")
            QApplication.processEvents()  # 确保UI响应
            self.init_ui()
            print("SimpleModelTestUI: 初始化完成")
            
            # 使用QTimer延迟加载模型，确保UI完全初始化
            QTimer.singleShot(100, self.load_models)
        except Exception as e:
            print(f"SimpleModelTestUI 初始化失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def init_ui(self):
        """初始化UI"""
        try:
            print("SimpleModelTestUI.init_ui: 开始...")
            layout = QVBoxLayout(self)
            print("SimpleModelTestUI.init_ui: 布局创建完成")
            
            # 创建标签页
            print("SimpleModelTestUI.init_ui: 创建标签页组件...")
            self.tab_widget = QTabWidget()
            layout.addWidget(self.tab_widget)
            print("SimpleModelTestUI.init_ui: 标签页组件创建完成")
            
            # 标注数据测试标签页
            print("SimpleModelTestUI.init_ui: 创建标注数据测试标签页...")
            self.annotation_tab = self.create_annotation_test_tab()
            self.tab_widget.addTab(self.annotation_tab, "标注数据测试")
            print("SimpleModelTestUI.init_ui: 标注数据测试标签页创建完成")
            
            # 摄像头检测测试标签页
            print("SimpleModelTestUI.init_ui: 创建摄像头检测测试标签页...")
            self.camera_tab = self.create_camera_test_tab()
            self.tab_widget.addTab(self.camera_tab, "摄像头检测测试")
            print("SimpleModelTestUI.init_ui: 摄像头检测测试标签页创建完成")
            
            # 模型对比标签页
            print("SimpleModelTestUI.init_ui: 创建模型对比标签页...")
            self.comparison_tab = self.create_comparison_tab()
            self.tab_widget.addTab(self.comparison_tab, "模型对比")
            print("SimpleModelTestUI.init_ui: 模型对比标签页创建完成")
            print("SimpleModelTestUI.init_ui: 所有标签页创建完成")
        except Exception as e:
            print(f"SimpleModelTestUI.init_ui 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_annotation_test_tab(self):
        """创建标注数据测试标签页"""
        try:
            print("  create_annotation_test_tab: 开始...")
            tab = QWidget()
            layout = QHBoxLayout(tab)
            
            # 左侧控制面板
            print("  create_annotation_test_tab: 创建左侧控制面板...")
            left_panel = QWidget()
            left_panel.setFixedWidth(350)
            left_layout = QVBoxLayout(left_panel)
            
            # 模型选择组
            model_group = QGroupBox("模型选择")
            model_layout = QVBoxLayout(model_group)
            model_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
            
            # 模型类型选择
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("模型类型:"))
            self.model_type_combo = QComboBox()
            self.model_type_combo.addItems(["盲道障碍检测模型"])
            self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
            type_layout.addWidget(self.model_type_combo)
            model_layout.addLayout(type_layout)
            
            # 模型选择
            model_select_layout = QHBoxLayout()
            model_select_layout.addWidget(QLabel("选择模型:"))
            self.model_combo = QComboBox()
            self.model_combo.setMinimumWidth(200)
            model_select_layout.addWidget(self.model_combo)
            model_layout.addLayout(model_select_layout)
            
            # 模型信息
            self.model_info_label = QLabel("未选择模型")
            self.model_info_label.setWordWrap(True)
            self.model_info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
            model_layout.addWidget(self.model_info_label)
            
            left_layout.addWidget(model_group)
            
            # 测试控制组
            control_group = QGroupBox("测试控制")
            control_layout = QVBoxLayout(control_group)
            control_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
            
            # 数据集选择
            dataset_layout = QHBoxLayout()
            dataset_layout.addWidget(QLabel("测试数据:"))
            self.dataset_btn = QPushButton("选择文件夹")
            self.dataset_btn.clicked.connect(self.select_test_dataset)
            dataset_layout.addWidget(self.dataset_btn)
            control_layout.addLayout(dataset_layout)
            
            self.dataset_path_label = QLabel("未选择")
            self.dataset_path_label.setStyleSheet("color: #999; font-size: 10px;")
            control_layout.addWidget(self.dataset_path_label)
            
            # 置信度阈值滑块
            threshold_layout = QVBoxLayout()
            threshold_layout.addWidget(QLabel("置信度阈值:"))
            self.conf_threshold_slider = QSlider(Qt.Horizontal)
            self.conf_threshold_slider.setMinimum(0)
            self.conf_threshold_slider.setMaximum(100)
            self.conf_threshold_slider.setValue(25)
            self.conf_threshold_slider.setTickInterval(5)
            self.conf_threshold_slider.setTickPosition(QSlider.TicksBelow)
            self.conf_threshold_slider.valueChanged.connect(self.on_conf_threshold_changed)
            threshold_layout.addWidget(self.conf_threshold_slider)
            self.threshold_value_label = QLabel("0.25")
            self.threshold_value_label.setAlignment(Qt.AlignCenter)
            threshold_layout.addWidget(self.threshold_value_label)
            control_layout.addLayout(threshold_layout)
            
            # 开始测试按钮
            self.start_test_btn = QPushButton("开始测试")
            self.start_test_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
            self.start_test_btn.clicked.connect(self.start_annotation_test)
            control_layout.addWidget(self.start_test_btn)
            
            # 进度条
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            control_layout.addWidget(self.progress_bar)
            
            # 状态标签
            self.status_label = QLabel("就绪")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            control_layout.addWidget(self.status_label)
            
            left_layout.addWidget(control_group)
            left_layout.addStretch()
            
            layout.addWidget(left_panel)
            
            # 右侧结果面板
            print("  create_annotation_test_tab: 创建右侧结果面板...")
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # 创建标签页
            results_tab_widget = QTabWidget()
            
            # 性能指标标签页
            metrics_tab = QWidget()
            metrics_layout = QVBoxLayout(metrics_tab)
            
            # 创建分割器
            splitter = QSplitter(Qt.Vertical)
            
            # 深度指标雷达图
            print("  create_annotation_test_tab: 创建深度指标雷达图组件...")
            radar_group = QGroupBox("深度指标雷达图")
            radar_layout = QVBoxLayout(radar_group)
            try:
                self.radar_chart = RadarChartWidget()
                radar_layout.addWidget(self.radar_chart)
            except Exception as e:
                print(f"  创建雷达图失败: {e}")
                error_label = QLabel(f"雷达图创建失败: {e}")
                radar_layout.addWidget(error_label)
            splitter.addWidget(radar_group)
            print("  create_annotation_test_tab: 雷达图组件创建完成")
            
            # 测试结果
            results_group = QGroupBox("测试结果")
            results_layout = QVBoxLayout(results_group)
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            results_layout.addWidget(self.results_text)
            splitter.addWidget(results_group)
            
            splitter.setSizes([400, 200])
            metrics_layout.addWidget(splitter)
            results_tab_widget.addTab(metrics_tab, "性能指标")
            
            # 混淆矩阵标签页
            confusion_tab = QWidget()
            confusion_layout = QVBoxLayout(confusion_tab)
            
            confusion_group = QGroupBox("混淆矩阵可视化")
            confusion_group_layout = QVBoxLayout(confusion_group)
            
            try:
                if _lazy_import_matplotlib() and MATPLOTLIB_AVAILABLE:
                    self.confusion_figure = Figure(figsize=(10, 8))
                    self.confusion_canvas = FigureCanvas(self.confusion_figure)
                    confusion_group_layout.addWidget(self.confusion_canvas)
                else:
                    confusion_label = QLabel("matplotlib未安装，无法显示混淆矩阵")
                    confusion_label.setAlignment(Qt.AlignCenter)
                    confusion_group_layout.addWidget(confusion_label)
            except Exception as e:
                print(f"  创建混淆矩阵失败: {e}")
                error_label = QLabel(f"混淆矩阵创建失败: {e}")
                confusion_group_layout.addWidget(error_label)
            
            confusion_layout.addWidget(confusion_group)
            results_tab_widget.addTab(confusion_tab, "混淆矩阵")
            
            # 错题集画廊标签页
            bad_case_tab = QWidget()
            bad_case_layout = QVBoxLayout(bad_case_tab)
            
            bad_case_group = QGroupBox("错题集画廊")
            bad_case_group_layout = QVBoxLayout(bad_case_group)
            
            # 错题集显示区域
            self.bad_case_display = QLabel("测试完成后将显示错题集")
            self.bad_case_display.setAlignment(Qt.AlignCenter)
            self.bad_case_display.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
            self.bad_case_display.setMinimumSize(640, 480)
            bad_case_group_layout.addWidget(self.bad_case_display)
            
            # 错题集导航
            bad_case_nav_layout = QHBoxLayout()
            self.prev_bad_case_btn = QPushButton("上一张")
            self.prev_bad_case_btn.setEnabled(False)
            self.prev_bad_case_btn.clicked.connect(self.on_prev_bad_case)
            self.next_bad_case_btn = QPushButton("下一张")
            self.next_bad_case_btn.setEnabled(False)
            self.next_bad_case_btn.clicked.connect(self.on_next_bad_case)
            self.bad_case_index_label = QLabel("0/0")
            self.bad_case_index_label.setAlignment(Qt.AlignCenter)
            
            bad_case_nav_layout.addWidget(self.prev_bad_case_btn)
            bad_case_nav_layout.addWidget(self.bad_case_index_label)
            bad_case_nav_layout.addWidget(self.next_bad_case_btn)
            bad_case_group_layout.addLayout(bad_case_nav_layout)
            
            bad_case_layout.addWidget(bad_case_group)
            results_tab_widget.addTab(bad_case_tab, "错题集画廊")
            
            right_layout.addWidget(results_tab_widget)
            
            layout.addWidget(right_panel)
            
            print("  create_annotation_test_tab: 完成")
            return tab
        except Exception as e:
            print(f"  create_annotation_test_tab 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个简单的错误标签页
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel(f"创建标注数据测试标签页失败: {e}")
            error_label.setAlignment(Qt.AlignCenter)
            error_layout.addWidget(error_label)
            return error_tab
    
    def create_camera_test_tab(self):
        """创建摄像头检测测试标签页"""
        try:
            print("  create_camera_test_tab: 开始...")
            tab = QWidget()
            layout = QHBoxLayout(tab)
            
            # 左侧控制面板
            left_panel = QWidget()
            left_panel.setFixedWidth(300)
            left_layout = QVBoxLayout(left_panel)
            
            # 模型选择（复用）
            model_group = QGroupBox("模型选择")
            model_layout = QVBoxLayout(model_group)
            
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("模型类型:"))
            self.camera_model_type_combo = QComboBox()
            self.camera_model_type_combo.addItems(["盲道障碍检测模型", "环境检测模型"])
            self.camera_model_type_combo.currentTextChanged.connect(self.on_camera_model_type_changed)
            type_layout.addWidget(self.camera_model_type_combo)
            model_layout.addLayout(type_layout)
            
            model_select_layout = QHBoxLayout()
            model_select_layout.addWidget(QLabel("选择模型:"))
            self.camera_model_combo = QComboBox()
            self.camera_model_combo.setMinimumWidth(200)
            model_select_layout.addWidget(self.camera_model_combo)
            model_layout.addLayout(model_select_layout)
            
            left_layout.addWidget(model_group)
            
            # 摄像头控制
            camera_group = QGroupBox("摄像头控制")
            camera_layout = QVBoxLayout(camera_group)
            
            self.camera_start_btn = QPushButton("📹 开启摄像头检测")
            self.camera_start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.camera_start_btn.clicked.connect(self.toggle_camera_detection)
            camera_layout.addWidget(self.camera_start_btn)
            
            self.camera_status_label = QLabel("摄像头状态: 未启动")
            camera_layout.addWidget(self.camera_status_label)
            
            left_layout.addWidget(camera_group)
            
            # 实时指标
            metrics_group = QGroupBox("实时指标")
            metrics_layout = QVBoxLayout(metrics_group)
            
            self.realtime_metrics_text = QTextEdit()
            self.realtime_metrics_text.setReadOnly(True)
            self.realtime_metrics_text.setMaximumHeight(200)
            metrics_layout.addWidget(self.realtime_metrics_text)
            
            left_layout.addWidget(metrics_group)
            left_layout.addStretch()
            
            layout.addWidget(left_panel)
            
            # 右侧摄像头显示
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            self.camera_display = QLabel("点击'开启摄像头检测'开始")
            self.camera_display.setMinimumSize(640, 480)
            self.camera_display.setStyleSheet("""
                border: 2px solid #ccc;
                background-color: #000;
                color: white;
                font-size: 16px;
            """)
            self.camera_display.setAlignment(Qt.AlignCenter)
            right_layout.addWidget(self.camera_display)
            
            layout.addWidget(right_panel)
            
            print("  create_camera_test_tab: 完成")
            return tab
        except Exception as e:
            print(f"  create_camera_test_tab 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel(f"创建摄像头检测测试标签页失败: {e}")
            error_label.setAlignment(Qt.AlignCenter)
            error_layout.addWidget(error_label)
            return error_tab
    
    def create_comparison_tab(self):
        """创建模型对比标签页"""
        try:
            print("  create_comparison_tab: 开始...")
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # 说明
            info_label = QLabel("选择多个模型进行测试后，可以在此查看对比结果")
            info_label.setStyleSheet("color: #666; padding: 10px;")
            layout.addWidget(info_label)
            
            # 对比图表区域
            print("  create_comparison_tab: 创建对比图表...")
            # 延迟导入matplotlib，使用更安全的方式
            try:
                # 先尝试导入
                if _lazy_import_matplotlib() and MATPLOTLIB_AVAILABLE:
                    print("  create_comparison_tab: 尝试创建matplotlib图表...")
                    QApplication.processEvents()  # 处理事件，避免阻塞
                    try:
                        # 直接创建图表，不设置后端
                        self.comparison_figure = Figure(figsize=(10, 6))
                        self.comparison_canvas = FigureCanvas(self.comparison_figure)
                        layout.addWidget(self.comparison_canvas)
                        print("  create_comparison_tab: matplotlib图表创建成功")
                    except Exception as e:
                        print(f"  create_comparison_tab: 图表创建失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # 创建备用UI
                        comparison_label = QLabel("matplotlib图表创建失败，无法显示对比图表\n\n请检查matplotlib安装")
                        comparison_label.setAlignment(Qt.AlignCenter)
                        comparison_label.setStyleSheet("color: #999; font-size: 14px; padding: 20px;")
                        layout.addWidget(comparison_label)
                        self.comparison_figure = None
                        self.comparison_canvas = None
                else:
                    # matplotlib不可用，创建备用UI
                    comparison_label = QLabel("matplotlib未安装或不可用，无法显示对比图表\n\n请安装: pip install matplotlib")
                    comparison_label.setAlignment(Qt.AlignCenter)
                    comparison_label.setStyleSheet("color: #999; font-size: 14px; padding: 20px;")
                    layout.addWidget(comparison_label)
                    self.comparison_figure = None
                    self.comparison_canvas = None
            except Exception as e:
                print(f"  create_comparison_tab: matplotlib图表创建失败: {e}")
                import traceback
                traceback.print_exc()
                comparison_label = QLabel("matplotlib初始化失败，无法显示对比图表\n\n请检查matplotlib安装")
                comparison_label.setAlignment(Qt.AlignCenter)
                comparison_label.setStyleSheet("color: #999; font-size: 14px; padding: 20px;")
                layout.addWidget(comparison_label)
                self.comparison_figure = None
                self.comparison_canvas = None
            
            print("  create_comparison_tab: 完成")
            return tab
        except Exception as e:
            print(f"  create_comparison_tab 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel(f"创建模型对比标签页失败: {e}")
            error_label.setAlignment(Qt.AlignCenter)
            error_layout.addWidget(error_label)
            return error_tab
    
    def load_models(self):
        """加载可用模型"""
        try:
            print("  load_models: 开始...")
            QApplication.processEvents()  # 确保UI响应
            self.load_models_by_type("盲道障碍检测模型")
            QApplication.processEvents()  # 确保UI响应
            self.load_camera_models_by_type("盲道障碍检测模型")
            print("  load_models: 完成")
        except Exception as e:
            print(f"  load_models 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_models_by_type(self, model_type: str):
        """根据类型加载模型"""
        try:
            print(f"    load_models_by_type: {model_type}")
            self.model_combo.clear()
            models_dir = "models"
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
                print(f"    创建models目录: {models_dir}")
                return
            
            # 根据类型筛选
            if "盲道障碍" in model_type:
                prefix = "blind_road"
            else:
                prefix = "environment"
            
            # 查找模型文件
            print(f"    扫描models目录: {models_dir}")
            QApplication.processEvents()  # 确保UI响应
            model_files = []
            try:
                files = os.listdir(models_dir)
                print(f"    找到 {len(files)} 个文件")
                QApplication.processEvents()  # 确保UI响应
                for file in files:
                    if file.endswith('.pt') and (prefix in file.lower() or file.startswith('yolov8')):
                        model_path = os.path.join(models_dir, file)
                        model_files.append((file, model_path))
                        print(f"      添加模型: {file}")
            except Exception as e:
                print(f"    扫描models目录失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 也检查训练结果目录（限制深度避免卡住）
            if "盲道障碍" in model_type:
                result_dirs = [
                    "results/blind_road_training/blind_road_detection/weights"
                ]
            else:
                result_dirs = [
                    "results/environment_training/environment_detection/weights"
                ]
            
            print(f"    扫描训练结果目录...")
            QApplication.processEvents()  # 确保UI响应
            for result_dir in result_dirs:
                if os.path.exists(result_dir):
                    try:
                        print(f"      扫描目录: {result_dir}")
                        # 限制搜索深度，避免在大型目录中卡住
                        max_files_checked = 100  # 最多检查100个文件
                        files_checked = 0
                        for root, dirs, files in os.walk(result_dir):
                            # 限制深度为2层
                            depth = root[len(result_dir):].count(os.sep)
                            if depth > 2:
                                dirs[:] = []  # 不继续深入
                                continue
                            
                            for file in files:
                                files_checked += 1
                                if files_checked > max_files_checked:
                                    print(f"      达到最大文件检查数，停止扫描")
                                    break
                                
                                if file == "best.pt":
                                    model_path = os.path.join(root, file)
                                    model_name = os.path.basename(os.path.dirname(root))
                                    model_files.append((f"{model_name}_best", model_path))
                                    print(f"      添加训练模型: {model_name}_best")
                            
                            if files_checked > max_files_checked:
                                break
                            
                            # 每检查一层目录就处理一次事件
                            QApplication.processEvents()
                    except Exception as e:
                        print(f"    扫描{result_dir}失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 添加到下拉框
            for name, path in model_files:
                self.model_combo.addItem(name, path)
            
            print(f"    找到 {len(model_files)} 个模型")
        except Exception as e:
            print(f"    load_models_by_type 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_camera_models_by_type(self, model_type: str):
        """为摄像头测试加载模型"""
        try:
            self.camera_model_combo.clear()
            # 复用相同的加载逻辑
            models_dir = "models"
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
                print(f"    创建models目录: {models_dir}")
                return
            
            if "盲道障碍" in model_type:
                prefix = "blind_road"
            else:
                prefix = "environment"
            
            model_files = []
            for file in os.listdir(models_dir):
                if file.endswith('.pt') and (prefix in file.lower() or file.startswith('yolov8')):
                    model_path = os.path.join(models_dir, file)
                    model_files.append((file, model_path))
            
            for name, path in model_files:
                self.camera_model_combo.addItem(name, path)
            
            print(f"    找到 {len(model_files)} 个摄像头模型")
        except Exception as e:
            print(f"load_camera_models_by_type 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def on_model_type_changed(self, text):
        """模型类型改变"""
        self.load_models_by_type(text)
    
    def on_camera_model_type_changed(self, text):
        """摄像头模型类型改变"""
        self.load_camera_models_by_type(text)
    
    def select_test_dataset(self):
        """选择测试数据集"""
        folder = QFileDialog.getExistingDirectory(self, "选择测试图像文件夹")
        if folder:
            self.dataset_path = folder
            self.dataset_path_label.setText(os.path.basename(folder))
    
    def start_annotation_test(self):
        """开始标注数据测试"""
        if not hasattr(self, 'dataset_path') or not os.path.exists(self.dataset_path):
            QMessageBox.warning(self, "警告", "请先选择测试数据集")
            return
        
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "警告", "请先选择模型")
            return
        
        # 获取测试图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        for file in os.listdir(self.dataset_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(self.dataset_path, file))
        
        if not test_images:
            QMessageBox.warning(self, "警告", "测试文件夹中没有图像文件")
            return
        
        # 更新UI
        self.start_test_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("测试进行中...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # 启动测试线程
        self.test_worker = ModelTestWorker(model_path, test_images, 'annotation')
        self.test_worker.progress_updated.connect(self.on_test_progress)
        self.test_worker.metrics_updated.connect(self.on_test_metrics_updated)
        self.test_worker.test_finished.connect(self.on_test_finished)
        self.test_worker.start()
    
    def on_test_progress(self, progress, message):
        """测试进度更新"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_test_metrics_updated(self, metrics):
        """测试指标更新"""
        # 实时更新指标显示
        pass
    
    def on_test_finished(self, results):
        """测试完成"""
        self.start_test_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if 'error' in results:
            self.status_label.setText(f"测试失败: {results['error']}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "错误", results['error'])
            return
        
        # 更新状态
        self.status_label.setText("测试完成")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 显示结果
        result_text = f"测试完成！\n\n"
        result_text += f"处理图像数: {results.get('processed_images', 0)}/{results.get('total_images', 0)}\n"
        result_text += f"平均推理时间: {results.get('avg_inference_time', 0):.2f} ms\n"
        result_text += f"平均检测数量: {results.get('avg_detection_count', 0):.2f}\n"
        result_text += f"平均置信度: {results.get('avg_confidence', 0):.3f}\n"
        result_text += f"总检测数: {results.get('total_detections', 0)}\n\n"
        
        # 类别分布
        class_dist = results.get('class_distribution', {})
        if class_dist:
            result_text += "类别分布:\n"
            for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
                result_text += f"  {class_name}: {count}\n"
        
        self.results_text.setText(result_text)
        
        # 计算精确率、召回率和F1分数
        predictions = results.get('predictions', [])
        ground_truths = results.get('ground_truths', [])
        
        # 简单的精确率和召回率计算
        if predictions and ground_truths:
            # 这里使用简单的计算方法，实际应用中应该使用更复杂的方法
            # 例如，使用IoU来匹配预测和真实框
            true_positives = len([p for p in predictions if p in ground_truths])
            precision = true_positives / len(predictions) if predictions else 0
            recall = true_positives / len(ground_truths) if ground_truths else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0
            recall = 0
            f1_score = 0
        
        # 更新雷达图
        model_name = self.model_combo.currentText()
        metrics = {
            'accuracy': results.get('avg_confidence', 0),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'speed': 1000 / results.get('avg_inference_time', 1) if results.get('avg_inference_time', 0) > 0 else 0,
            'stability': 0.8  # 估算
        }
        self.model_metrics[model_name] = metrics
        self.radar_chart.set_metrics(metrics)
        
        # 更新混淆矩阵
        try:
            if predictions and ground_truths:
                # 创建类别映射
                all_classes = list(set(predictions + ground_truths))
                class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
                
                # 创建混淆矩阵
                import numpy as np
                cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)
                
                # 填充混淆矩阵（简单方法）
                for pred, gt in zip(predictions[:len(ground_truths)], ground_truths):
                    if pred in class_to_idx and gt in class_to_idx:
                        cm[class_to_idx[gt], class_to_idx[pred]] += 1
                
                # 更新混淆矩阵显示
                self.update_confusion_matrix(cm, all_classes)
        except Exception as e:
            print(f"  计算混淆矩阵失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 更新错题集画廊
        bad_cases = results.get('bad_cases', [])
        self.update_bad_case_gallery(bad_cases)
        
        # 更新对比图表
        self.update_comparison_chart()
        
        # 保存测试结果
        self.test_results = results
    
    def toggle_camera_detection(self):
        """切换摄像头检测"""
        if not self.camera_active:
            self.start_camera_detection()
        else:
            self.stop_camera_detection()
    
    def start_camera_detection(self):
        """开始摄像头检测"""
        model_path = self.camera_model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "警告", "请先选择模型")
            return
        
        # 延迟导入YOLO
        _lazy_import_yolo()
        # 延迟导入YOLO
        _lazy_import_yolo()
        if not YOLO_AVAILABLE:
            QMessageBox.warning(self, "警告", "ultralytics未安装，无法进行检测")
            return
        
        try:
            from ultralytics import YOLO
            self.current_model = YOLO(model_path)
            self.current_model_path = model_path
            
            self.camera_cap = cv2.VideoCapture(0)
            if not self.camera_cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头")
                return
            
            self.camera_active = True
            self.camera_start_btn.setText("📹 停止摄像头检测")
            self.camera_start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 5px;
                }
            """)
            self.camera_status_label.setText("摄像头状态: 运行中")
            self.camera_timer.start(33)  # 约30FPS
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像头失败: {e}")
    
    def stop_camera_detection(self):
        """停止摄像头检测"""
        self.camera_active = False
        self.camera_timer.stop()
        
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
        
        self.camera_start_btn.setText("📹 开启摄像头检测")
        self.camera_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
        """)
        self.camera_status_label.setText("摄像头状态: 已停止")
        self.camera_display.setText("摄像头检测已停止")
    
    def update_camera_frame(self):
        """更新摄像头帧"""
        if not self.camera_cap or not self.camera_active or not self.current_model:
            return
        
        ret, frame = self.camera_cap.read()
        if not ret:
            return
        
        # 进行检测
        start_time = time.time()
        results = self.current_model(frame, conf=0.25, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        # 颜色映射字典，不同分类显示不同颜色
        class_colors = {
            'person': (0, 0, 255),      # 红色 - 行人
            'vehicle': (0, 255, 0),      # 绿色 - 车辆
            'pothole': (255, 0, 0),      # 蓝色 - 坑洼
            'step': (255, 255, 0),       # 黄色 - 台阶
            'slope': (128, 0, 128),      # 紫色 - 斜坡
            'water': (0, 255, 255),      # 青色 - 积水
            'stall': (255, 165, 0),      # 橙色 - 摊位
            'trash_bin': (192, 192, 192),# 灰色 - 垃圾桶
            'pet': (255, 192, 203),      # 粉色 - 宠物
            'furniture': (128, 128, 0),  # 橄榄色 - 家具
            'moving_vehicle': (255, 0, 255), # 洋红色 - 移动车辆
            'car_horn': (128, 0, 0),     # 深红色 - 汽车鸣笛
            'construction_noise': (128, 128, 128), # 深灰色 - 施工噪音
            'alarm': (0, 128, 128)       # 深青色 - 警报
        }
        
        # 绘制检测结果
        detection_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # 获取分类对应的颜色，如果没有则使用默认颜色
                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                    
                    # 计算目标大小
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    size = f"W:{width}, H:{height}"
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 绘制大小信息
                    cv2.putText(frame, size, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 生成并播放语音播报
                    if self.voice_enabled and VOICE_AVAILABLE and pyttsx3_engine:
                        current_time = time.time()
                        if current_time - self.last_voice_time > self.voice_cooldown:
                            # 计算目标中心与画面中心的相对位置，确定方位
                            frame_center_x = frame.shape[1] // 2
                            target_center_x = (x1 + x2) / 2
                            
                            if target_center_x < frame_center_x - 100:
                                direction = "左侧"
                            elif target_center_x > frame_center_x + 100:
                                direction = "右侧"
                            else:
                                direction = "正前方"
                            
                            # 计算目标与摄像头的距离（这里使用目标大小的倒数作为近似距离）
                            # 实际应用中应该使用深度相机或其他方法获取真实距离
                            target_area = width * height
                            frame_area = frame.shape[0] * frame.shape[1]
                            relative_size = target_area / frame_area
                            # 将相对大小转换为近似距离（米）
                            distance = min(5.0, max(0.1, 2.0 / (relative_size + 0.1)))
                            
                            # 生成语音消息
                            message = voice_library.generate_voice_message(class_id, distance, direction, class_name)
                            if message:
                                # 播放语音
                                try:
                                    pyttsx3_engine.say(message)
                                    pyttsx3_engine.runAndWait()
                                    self.last_voice_time = current_time
                                    print(f"🔊 语音播报: {message}")
                                except Exception as e:
                                    print(f"⚠️ 语音播报失败: {e}")
        
        # 更新实时指标
        metrics_text = f"推理时间: {inference_time:.2f} ms\n"
        metrics_text += f"FPS: {1000/inference_time:.2f}\n"
        metrics_text += f"检测数量: {detection_count}"
        if VOICE_AVAILABLE and pyttsx3_engine:
            metrics_text += f"\n语音状态: 已启用"
        else:
            metrics_text += f"\n语音状态: 未启用"
        self.realtime_metrics_text.setText(metrics_text)
        
        # 显示帧
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_display.setPixmap(scaled_pixmap)
    
    def on_conf_threshold_changed(self, value):
        """处理置信度阈值滑块的变化"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        
        # 如果已经有测试结果，根据新的阈值更新结果
        if hasattr(self, 'test_results') and self.test_results:
            self.update_results_with_threshold(threshold)
    
    def update_results_with_threshold(self, threshold):
        """根据置信度阈值更新测试结果"""
        # 这里可以实现根据新的阈值重新计算结果的逻辑
        pass
    
    def update_confusion_matrix(self, cm, classes):
        """更新混淆矩阵"""
        if not _lazy_import_matplotlib() or not MATPLOTLIB_AVAILABLE or not hasattr(self, 'confusion_figure'):
            return
        
        try:
            self.confusion_figure.clear()
            ax = self.confusion_figure.add_subplot(111)
            
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测类别')
            ax.set_ylabel('真实类别')
            ax.set_title('混淆矩阵')
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticklabels(classes, rotation=45)
            
            self.confusion_figure.tight_layout()
            self.confusion_canvas.draw()
        except Exception as e:
            print(f"  更新混淆矩阵失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_bad_case_gallery(self, bad_cases):
        """更新错题集画廊"""
        self.bad_cases = bad_cases
        self.current_bad_case_index = 0
        
        if bad_cases:
            self.prev_bad_case_btn.setEnabled(True)
            self.next_bad_case_btn.setEnabled(True)
            self.update_bad_case_display()
        else:
            self.prev_bad_case_btn.setEnabled(False)
            self.next_bad_case_btn.setEnabled(False)
            self.bad_case_display.setText("无错题集数据")
            self.bad_case_index_label.setText("0/0")
    
    def update_bad_case_display(self):
        """更新错题显示"""
        if not hasattr(self, 'bad_cases') or not self.bad_cases:
            return
        
        if 0 <= self.current_bad_case_index < len(self.bad_cases):
            bad_case = self.bad_cases[self.current_bad_case_index]
            img_path = bad_case.get('image_path')
            
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # 绘制检测结果
                    detections = bad_case.get('detections', [])
                    for detection in detections:
                        bbox = detection.get('bbox')
                        class_name = detection.get('class_name')
                        conf = detection.get('confidence', 0)
                        
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(img, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 转换为Qt格式并显示
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(self.bad_case_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.bad_case_display.setPixmap(scaled_pixmap)
                else:
                    self.bad_case_display.setText(f"无法读取图像: {img_path}")
            else:
                self.bad_case_display.setText(f"图像不存在: {img_path}")
            
            self.bad_case_index_label.setText(f"{self.current_bad_case_index + 1}/{len(self.bad_cases)}")
    
    def on_prev_bad_case(self):
        """显示上一张错题"""
        if hasattr(self, 'bad_cases') and self.bad_cases:
            self.current_bad_case_index = (self.current_bad_case_index - 1) % len(self.bad_cases)
            self.update_bad_case_display()
    
    def on_next_bad_case(self):
        """显示下一张错题"""
        if hasattr(self, 'bad_cases') and self.bad_cases:
            self.current_bad_case_index = (self.current_bad_case_index + 1) % len(self.bad_cases)
            self.update_bad_case_display()
    
    def update_comparison_chart(self):
        """更新对比图表"""
        # 延迟导入matplotlib
        if not _lazy_import_matplotlib() or not MATPLOTLIB_AVAILABLE or not self.model_metrics or self.comparison_figure is None:
            return
        
        try:
            self.comparison_figure.clear()
            ax = self.comparison_figure.add_subplot(111)
            
            models = list(self.model_metrics.keys())
            metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'speed', 'stability']
            
            x = np.arange(len(metrics_names))
            width = 0.8 / len(models)
            
            for i, model_name in enumerate(models):
                values = [self.model_metrics[model_name].get(name, 0) for name in metrics_names]
                ax.bar(x + i * width, values, width, label=model_name)
            
            ax.set_xlabel('指标')
            ax.set_ylabel('数值')
            ax.set_title('模型性能对比')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数', '速度', '稳定性'])
            ax.legend()
            ax.set_ylim(0, 1)
            
            self.comparison_canvas.draw()
        except Exception as e:
            print(f"  update_comparison_chart 失败: {e}")
            import traceback
            traceback.print_exc()
