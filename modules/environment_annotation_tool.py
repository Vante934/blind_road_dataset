#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测标注工具
支持24类环境物体的标注，用于训练深度学习模型
"""

import sys
import os
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QComboBox, QListWidget, 
                             QTextEdit, QFileDialog, QMessageBox, QGroupBox, 
                             QSlider, QSpinBox, QCheckBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont, QColor

class EnvironmentAnnotationTool(QMainWindow):
    """环境检测标注工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("环境检测标注工具 v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # 环境物体类别定义
        self.environment_classes = {
            0: {"name": "晴天", "color": (255, 255, 0), "description": "阳光充足，视野清晰"},
            1: {"name": "雨天", "color": (0, 255, 255), "description": "下雨天气，路面湿滑"},
            2: {"name": "雪天", "color": (255, 255, 255), "description": "下雪天气，路面可能结冰"},
            3: {"name": "雾天", "color": (200, 200, 200), "description": "雾气弥漫，能见度低"},
            4: {"name": "明亮", "color": (255, 255, 0), "description": "光照充足，视野良好"},
            5: {"name": "正常", "color": (128, 128, 128), "description": "光照正常，视野一般"},
            6: {"name": "昏暗", "color": (64, 64, 64), "description": "光照不足，视野受限"},
            7: {"name": "黑暗", "color": (0, 0, 0), "description": "光照极差，视野严重受限"},
            8: {"name": "平整", "color": (0, 255, 0), "description": "路面平整，行走安全"},
            9: {"name": "湿滑", "color": (0, 0, 255), "description": "路面湿滑，行走需小心"},
            10: {"name": "结冰", "color": (255, 255, 255), "description": "路面结冰，行走危险"},
            11: {"name": "坑洼", "color": (128, 64, 0), "description": "路面坑洼，行走不便"},
            12: {"name": "施工", "color": (255, 0, 0), "description": "施工区域，行走危险"},
            13: {"name": "盲道", "color": (255, 165, 0), "description": "盲道设施，行走安全"},
            14: {"name": "人行道", "color": (0, 128, 0), "description": "人行道区域，行走安全"},
            15: {"name": "路口", "color": (255, 0, 255), "description": "交通路口，行走需注意"},
            16: {"name": "施工区", "color": (255, 0, 0), "description": "施工区域，行走危险"},
            17: {"name": "停车场", "color": (128, 128, 128), "description": "停车场区域，行走需注意"},
            18: {"name": "护栏", "color": (0, 0, 128), "description": "安全护栏，行走安全"},
            19: {"name": "警示牌", "color": (255, 255, 0), "description": "警示标志，行走需注意"},
            20: {"name": "红绿灯", "color": (0, 255, 0), "description": "交通信号灯，行走需注意"},
            21: {"name": "斑马线", "color": (255, 255, 255), "description": "人行横道，行走安全"},
            22: {"name": "无障碍设施", "color": (0, 255, 255), "description": "无障碍设施，行走安全"},
            23: {"name": "其他", "color": (128, 128, 128), "description": "其他环境物体"}
        }
        
        # 标注数据
        self.current_image = None
        self.current_image_path = ""
        self.annotations = []
        self.current_class = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_rect = None
        
        # 文件管理
        self.image_files = []
        self.current_image_index = 0
        self.annotation_dir = "data/environment_annotations"
        self.image_dir = "data/images"
        
        # 创建目录
        os.makedirs(self.annotation_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.init_ui()
        self.load_image_list()
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 中间图像显示区域
        image_panel = self.create_image_panel()
        splitter.addWidget(image_panel)
        
        # 右侧标注信息面板
        right_panel = self.create_annotation_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([300, 800, 300])
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        # 选择图像目录
        self.select_dir_btn = QPushButton("选择图像目录")
        self.select_dir_btn.clicked.connect(self.select_image_directory)
        file_layout.addWidget(self.select_dir_btn)
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_selected_image)
        file_layout.addWidget(QLabel("图像列表:"))
        file_layout.addWidget(self.image_list)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        file_layout.addLayout(nav_layout)
        
        layout.addWidget(file_group)
        
        # 标注工具组
        tool_group = QGroupBox("标注工具")
        tool_layout = QVBoxLayout(tool_group)
        
        # 类别选择
        tool_layout.addWidget(QLabel("选择环境类别:"))
        self.class_combo = QComboBox()
        for class_id, class_info in self.environment_classes.items():
            self.class_combo.addItem(f"{class_id}: {class_info['name']}")
        self.class_combo.currentIndexChanged.connect(self.change_class)
        tool_layout.addWidget(self.class_combo)
        
        # 标注模式
        tool_layout.addWidget(QLabel("标注模式:"))
        self.annotation_mode = QComboBox()
        self.annotation_mode.addItems(["矩形框", "多边形", "点标注"])
        tool_layout.addWidget(self.annotation_mode)
        
        # 工具按钮
        self.start_annotation_btn = QPushButton("开始标注")
        self.start_annotation_btn.clicked.connect(self.start_annotation)
        tool_layout.addWidget(self.start_annotation_btn)
        
        self.finish_annotation_btn = QPushButton("完成标注")
        self.finish_annotation_btn.clicked.connect(self.finish_annotation)
        self.finish_annotation_btn.setEnabled(False)
        tool_layout.addWidget(self.finish_annotation_btn)
        
        self.clear_annotation_btn = QPushButton("清除当前标注")
        self.clear_annotation_btn.clicked.connect(self.clear_current_annotation)
        tool_layout.addWidget(self.clear_annotation_btn)
        
        layout.addWidget(tool_group)
        
        # 保存和加载组
        save_group = QGroupBox("保存和加载")
        save_layout = QVBoxLayout(save_group)
        
        self.save_btn = QPushButton("保存标注")
        self.save_btn.clicked.connect(self.save_annotations)
        save_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("加载标注")
        self.load_btn.clicked.connect(self.load_annotations)
        save_layout.addWidget(self.load_btn)
        
        self.export_btn = QPushButton("导出训练数据")
        self.export_btn.clicked.connect(self.export_training_data)
        save_layout.addWidget(self.export_btn)
        
        layout.addWidget(save_group)
        
        # 统计信息组
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        return panel
        
    def create_image_panel(self):
        """创建中间图像显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        layout.addWidget(self.image_label)
        
        # 图像信息
        self.image_info_label = QLabel("请选择图像开始标注")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_info_label)
        
        return panel
        
    def create_annotation_panel(self):
        """创建右侧标注信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 当前标注组
        current_group = QGroupBox("当前标注")
        current_layout = QVBoxLayout(current_group)
        
        self.current_annotation_text = QTextEdit()
        self.current_annotation_text.setMaximumHeight(200)
        self.current_annotation_text.setReadOnly(True)
        current_layout.addWidget(self.current_annotation_text)
        
        layout.addWidget(current_group)
        
        # 标注列表组
        list_group = QGroupBox("标注列表")
        list_layout = QVBoxLayout(list_group)
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.select_annotation)
        list_layout.addWidget(self.annotation_list)
        
        # 标注操作按钮
        list_btn_layout = QHBoxLayout()
        self.edit_annotation_btn = QPushButton("编辑")
        self.edit_annotation_btn.clicked.connect(self.edit_annotation)
        self.delete_annotation_btn = QPushButton("删除")
        self.delete_annotation_btn.clicked.connect(self.delete_annotation)
        list_btn_layout.addWidget(self.edit_annotation_btn)
        list_btn_layout.addWidget(self.delete_annotation_btn)
        list_layout.addLayout(list_btn_layout)
        
        layout.addWidget(list_group)
        
        # 类别信息组
        class_group = QGroupBox("类别信息")
        class_layout = QVBoxLayout(class_group)
        
        self.class_info_text = QTextEdit()
        self.class_info_text.setMaximumHeight(150)
        self.class_info_text.setReadOnly(True)
        class_layout.addWidget(self.class_info_text)
        
        layout.addWidget(class_group)
        
        # 更新类别信息
        self.update_class_info()
        
        return panel
        
    def select_image_directory(self):
        """选择图像目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if directory:
            self.image_dir = directory
            self.load_image_list()
            
    def load_image_list(self):
        """加载图像列表"""
        self.image_list.clear()
        self.image_files = []
        
        if os.path.exists(self.image_dir):
            for file in os.listdir(self.image_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_files.append(file)
                    self.image_list.addItem(file)
                    
        self.update_stats()
        
    def load_selected_image(self, item):
        """加载选中的图像"""
        if item:
            filename = item.text()
            self.current_image_index = self.image_files.index(filename)
            self.load_image(filename)
            
    def load_image(self, filename):
        """加载图像"""
        image_path = os.path.join(self.image_dir, filename)
        if os.path.exists(image_path):
            self.current_image_path = image_path
            self.current_image = cv2.imread(image_path)
            self.annotations = []
            self.load_annotations_for_image(filename)
            self.display_image()
            self.update_image_info()
            self.update_annotation_list()
            
    def display_image(self):
        """显示图像"""
        if self.current_image is not None:
            # 调整图像大小以适应显示
            height, width = self.current_image.shape[:2]
            max_width = 800
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.current_image = cv2.resize(self.current_image, (new_width, new_height))
                
            # 绘制标注
            display_image = self.current_image.copy()
            for annotation in self.annotations:
                self.draw_annotation(display_image, annotation)
                
            # 绘制当前正在标注的矩形
            if self.temp_rect:
                self.draw_annotation(display_image, self.temp_rect)
                
            # 转换为QImage并显示
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)
            
    def draw_annotation(self, image, annotation):
        """绘制标注"""
        if 'bbox' in annotation:
            x1, y1, x2, y2 = annotation['bbox']
            class_id = annotation['class_id']
            class_info = self.environment_classes[class_id]
            color = class_info['color']
            
            # 绘制矩形框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制类别标签
            label = f"{class_id}: {class_info['name']}"
            cv2.putText(image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
    def mouse_press_event(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.annotation_mode.currentText() == "矩形框":
            self.drawing = True
            self.start_point = (event.x(), event.y())
            self.temp_rect = None
            
    def mouse_move_event(self, event):
        """鼠标移动事件"""
        if self.drawing and self.start_point:
            self.end_point = (event.x(), event.y())
            self.temp_rect = {
                'bbox': [self.start_point[0], self.start_point[1], 
                        self.end_point[0], self.end_point[1]],
                'class_id': self.current_class
            }
            self.display_image()
            
    def mouse_release_event(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_rect:
                self.annotations.append(self.temp_rect.copy())
                self.temp_rect = None
                self.update_annotation_list()
                self.display_image()
                
    def change_class(self, index):
        """改变标注类别"""
        self.current_class = index
        self.update_class_info()
        
    def update_class_info(self):
        """更新类别信息显示"""
        class_info = self.environment_classes[self.current_class]
        info_text = f"类别ID: {self.current_class}\n"
        info_text += f"类别名称: {class_info['name']}\n"
        info_text += f"描述: {class_info['description']}\n"
        info_text += f"颜色: {class_info['color']}"
        self.class_info_text.setPlainText(info_text)
        
    def start_annotation(self):
        """开始标注"""
        if self.current_image is not None:
            self.start_annotation_btn.setEnabled(False)
            self.finish_annotation_btn.setEnabled(True)
            self.image_label.setCursor(Qt.CrossCursor)
            
    def finish_annotation(self):
        """完成标注"""
        self.start_annotation_btn.setEnabled(True)
        self.finish_annotation_btn.setEnabled(False)
        self.image_label.setCursor(Qt.ArrowCursor)
        self.temp_rect = None
        self.display_image()
        
    def clear_current_annotation(self):
        """清除当前标注"""
        self.annotations = []
        self.temp_rect = None
        self.update_annotation_list()
        self.display_image()
        
    def update_annotation_list(self):
        """更新标注列表"""
        self.annotation_list.clear()
        for i, annotation in enumerate(self.annotations):
            class_info = self.environment_classes[annotation['class_id']]
            item_text = f"{i+1}. {class_info['name']} ({annotation['bbox']})"
            self.annotation_list.addItem(item_text)
            
    def select_annotation(self, item):
        """选择标注"""
        if item:
            index = self.annotation_list.row(item)
            if 0 <= index < len(self.annotations):
                annotation = self.annotations[index]
                info_text = f"标注 {index+1}:\n"
                info_text += f"类别: {self.environment_classes[annotation['class_id']]['name']}\n"
                info_text += f"边界框: {annotation['bbox']}\n"
                info_text += f"描述: {self.environment_classes[annotation['class_id']]['description']}"
                self.current_annotation_text.setPlainText(info_text)
                
    def edit_annotation(self):
        """编辑标注"""
        current_item = self.annotation_list.currentItem()
        if current_item:
            index = self.annotation_list.row(current_item)
            if 0 <= index < len(self.annotations):
                # 这里可以添加编辑对话框
                QMessageBox.information(self, "编辑标注", "编辑功能正在开发中...")
                
    def delete_annotation(self):
        """删除标注"""
        current_item = self.annotation_list.currentItem()
        if current_item:
            index = self.annotation_list.row(current_item)
            if 0 <= index < len(self.annotations):
                del self.annotations[index]
                self.update_annotation_list()
                self.display_image()
                
    def save_annotations(self):
        """保存标注"""
        if self.current_image_path and self.annotations:
            filename = os.path.basename(self.current_image_path)
            annotation_filename = os.path.splitext(filename)[0] + '.json'
            annotation_path = os.path.join(self.annotation_dir, annotation_filename)
            
            annotation_data = {
                'image_path': self.current_image_path,
                'image_filename': filename,
                'annotations': self.annotations,
                'total_annotations': len(self.annotations)
            }
            
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
                
            QMessageBox.information(self, "保存成功", f"标注已保存到: {annotation_path}")
            self.update_stats()
            
    def load_annotations(self):
        """加载标注"""
        if self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            annotation_filename = os.path.splitext(filename)[0] + '.json'
            annotation_path = os.path.join(self.annotation_dir, annotation_filename)
            
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    self.annotations = annotation_data.get('annotations', [])
                    self.update_annotation_list()
                    self.display_image()
                    QMessageBox.information(self, "加载成功", "标注已加载")
            else:
                QMessageBox.warning(self, "文件不存在", "未找到对应的标注文件")
                
    def load_annotations_for_image(self, filename):
        """为图像加载标注"""
        annotation_filename = os.path.splitext(filename)[0] + '.json'
        annotation_path = os.path.join(self.annotation_dir, annotation_filename)
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
                self.annotations = annotation_data.get('annotations', [])
                
    def export_training_data(self):
        """导出训练数据"""
        # 收集所有标注数据
        all_annotations = []
        for filename in self.image_files:
            annotation_filename = os.path.splitext(filename)[0] + '.json'
            annotation_path = os.path.join(self.annotation_dir, annotation_filename)
            
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    all_annotations.append(annotation_data)
                    
        if all_annotations:
            # 导出为YOLO格式
            self.export_yolo_format(all_annotations)
            QMessageBox.information(self, "导出成功", "训练数据已导出为YOLO格式")
        else:
            QMessageBox.warning(self, "导出失败", "没有找到标注数据")
            
    def export_yolo_format(self, all_annotations):
        """导出为YOLO格式"""
        # 创建YOLO格式的标注文件
        yolo_dir = "data/yolo_environment_dataset"
        os.makedirs(yolo_dir, exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels"), exist_ok=True)
        
        # 创建类别文件
        classes_file = os.path.join(yolo_dir, "classes.txt")
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_id, class_info in self.environment_classes.items():
                f.write(f"{class_info['name']}\n")
                
        # 转换标注格式
        for annotation_data in all_annotations:
            image_path = annotation_data['image_path']
            annotations = annotation_data['annotations']
            
            # 复制图像
            image_filename = os.path.basename(image_path)
            yolo_image_path = os.path.join(yolo_dir, "images", image_filename)
            if os.path.exists(image_path):
                import shutil
                shutil.copy2(image_path, yolo_image_path)
                
            # 创建YOLO格式的标注文件
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(yolo_dir, "labels", label_filename)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    if 'bbox' in annotation:
                        x1, y1, x2, y2 = annotation['bbox']
                        class_id = annotation['class_id']
                        
                        # 转换为YOLO格式 (归一化坐标)
                        # 这里需要获取原始图像尺寸
                        if os.path.exists(image_path):
                            img = cv2.imread(image_path)
                            h, w = img.shape[:2]
                            
                            # 归一化坐标
                            x_center = (x1 + x2) / 2 / w
                            y_center = (y1 + y2) / 2 / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            
    def prev_image(self):
        """上一张图像"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            
    def next_image(self):
        """下一张图像"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
            
    def update_image_info(self):
        """更新图像信息"""
        if self.current_image is not None:
            filename = os.path.basename(self.current_image_path)
            info_text = f"图像: {filename}\n"
            info_text += f"尺寸: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n"
            info_text += f"标注数量: {len(self.annotations)}"
            self.image_info_label.setText(info_text)
            
    def update_stats(self):
        """更新统计信息"""
        total_images = len(self.image_files)
        annotated_images = 0
        total_annotations = 0
        
        for filename in self.image_files:
            annotation_filename = os.path.splitext(filename)[0] + '.json'
            annotation_path = os.path.join(self.annotation_dir, annotation_filename)
            
            if os.path.exists(annotation_path):
                annotated_images += 1
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    total_annotations += len(annotation_data.get('annotations', []))
                    
        stats_text = f"总图像数: {total_images}\n"
        stats_text += f"已标注图像: {annotated_images}\n"
        stats_text += f"总标注数: {total_annotations}\n"
        stats_text += f"标注进度: {annotated_images/total_images*100:.1f}%" if total_images > 0 else "标注进度: 0%"
        
        self.stats_text.setPlainText(stats_text)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = EnvironmentAnnotationTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()







