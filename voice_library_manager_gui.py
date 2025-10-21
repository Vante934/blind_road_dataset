#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音库管理GUI界面
支持四种文本提示类型：障碍类型、用户行进路线提示、动态障碍物轨迹预测提示、环境事物提示
支持手动输入数据集并自动分析，支持数据集网址解析
"""

import sys
import json
import os
import re
import urllib.parse
import urllib.request
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyttsx3
import threading
import time

class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_text_dataset(self, text_data):
        """分析文本数据集"""
        results = {
            'obstacle_types': [],
            'route_guidance': [],
            'trajectory_prediction': [],
            'environment_awareness': []
        }
        
        # 障碍类型关键词
        obstacle_keywords = [
            '车辆', '汽车', '摩托车', '自行车', '行人', '宠物', '垃圾桶', '摊位',
            '台阶', '坑洼', '积水', '斜坡', '栏杆', '柱子', '树木', '花坛',
            '施工', '障碍物', '阻挡', '阻挡物'
        ]
        
        # 路线提示关键词
        route_keywords = [
            '盲道', '直行', '左转', '右转', '转弯', '前进', '后退', '停止',
            '路径', '路线', '导航', '方向', '前方', '后方', '左侧', '右侧',
            '继续', '保持', '改变', '调整'
        ]
        
        # 轨迹预测关键词
        trajectory_keywords = [
            '预测', '轨迹', '移动', '接近', '远离', '速度', '方向', '路径',
            '碰撞', '避让', '等待', '通过', '交叉', '相遇', '跟随', '超越'
        ]
        
        # 环境感知关键词
        environment_keywords = [
            '天气', '晴天', '雨天', '雾天', '夜晚', '白天', '光线', '声音',
            '噪音', '鸣笛', '警报', '施工', '交通', '人流', '车流', '环境',
            '温度', '湿度', '风速', '能见度'
        ]
        
        # 分析文本内容
        for line in text_data.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 检查障碍类型
            for keyword in obstacle_keywords:
                if keyword in line:
                    results['obstacle_types'].append({
                        'text': line,
                        'keyword': keyword,
                        'category': self._categorize_obstacle(keyword)
                    })
                    break
            
            # 检查路线提示
            for keyword in route_keywords:
                if keyword in line:
                    results['route_guidance'].append({
                        'text': line,
                        'keyword': keyword,
                        'type': self._categorize_route(keyword)
                    })
                    break
            
            # 检查轨迹预测
            for keyword in trajectory_keywords:
                if keyword in line:
                    results['trajectory_prediction'].append({
                        'text': line,
                        'keyword': keyword,
                        'action': self._categorize_trajectory(keyword)
                    })
                    break
            
            # 检查环境感知
            for keyword in environment_keywords:
                if keyword in line:
                    results['environment_awareness'].append({
                        'text': line,
                        'keyword': keyword,
                        'condition': self._categorize_environment(keyword)
                    })
                    break
        
        return results
    
    def _categorize_obstacle(self, keyword):
        """分类障碍物"""
        if keyword in ['车辆', '汽车', '摩托车', '自行车']:
            return 'vehicle'
        elif keyword in ['行人', '宠物']:
            return 'person'
        elif keyword in ['台阶', '坑洼', '积水', '斜坡']:
            return 'ground_hazard'
        elif keyword in ['垃圾桶', '摊位', '栏杆', '柱子', '树木', '花坛']:
            return 'static_object'
        else:
            return 'other'
    
    def _categorize_route(self, keyword):
        """分类路线提示"""
        if keyword in ['直行', '前进', '继续']:
            return 'straight'
        elif keyword in ['左转', '右转', '转弯']:
            return 'turn'
        elif keyword in ['停止', '等待']:
            return 'stop'
        else:
            return 'guidance'
    
    def _categorize_trajectory(self, keyword):
        """分类轨迹预测"""
        if keyword in ['接近', '移动', '跟随']:
            return 'approaching'
        elif keyword in ['远离', '通过', '超越']:
            return 'departing'
        elif keyword in ['碰撞', '避让', '等待']:
            return 'avoidance'
        else:
            return 'prediction'
    
    def _categorize_environment(self, keyword):
        """分类环境条件"""
        if keyword in ['晴天', '雨天', '雾天', '夜晚', '白天']:
            return 'weather'
        elif keyword in ['声音', '噪音', '鸣笛', '警报']:
            return 'sound'
        elif keyword in ['光线', '能见度']:
            return 'visibility'
        else:
            return 'general'

class URLParser:
    """网址解析器"""
    
    def __init__(self):
        self.supported_domains = [
            'github.com', 'gitlab.com', 'kaggle.com', 'dataset.com',
            'data.gov', 'opendata', 'archive.org'
        ]
    
    def parse_url(self, url):
        """解析网址"""
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            result = {
                'url': url,
                'domain': domain,
                'is_supported': any(supported in domain for supported in self.supported_domains),
                'file_type': self._detect_file_type(url),
                'content': None
            }
            
            # 尝试获取内容
            if result['is_supported']:
                result['content'] = self._fetch_content(url)
            
            return result
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'is_supported': False
            }
    
    def _detect_file_type(self, url):
        """检测文件类型"""
        url_lower = url.lower()
        if url_lower.endswith('.json'):
            return 'json'
        elif url_lower.endswith('.csv'):
            return 'csv'
        elif url_lower.endswith('.txt'):
            return 'text'
        elif url_lower.endswith('.xml'):
            return 'xml'
        else:
            return 'unknown'
    
    def _fetch_content(self, url):
        """获取网址内容"""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            return f"获取内容失败: {e}"

class VoiceLibraryManagerGUI(QMainWindow):
    """语音库管理GUI界面"""
    
    def __init__(self):
        super().__init__()
        self.library_file = "voice_text_library.json"
        self.text_library = {}
        self.dataset_analyzer = DatasetAnalyzer()
        self.url_parser = URLParser()
        self.tts_engine = None
        self.init_ui()
        self.load_library()
        self.apply_memory()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("语音库管理系统 - 盲道检测项目")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 设置窗口可以自由调节大小
        self.setMinimumSize(1200, 800)
        self.setMaximumSize(2000, 1200)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                font-size: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                font-size: 16px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 18px;
                text-align: center;
                font-size: 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                font-size: 14px;
                padding: 8px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 使用QSplitter创建可调整大小的分区
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板 - 分类管理
        left_panel = self.create_left_panel()
        left_panel.setMinimumWidth(250)
        main_splitter.addWidget(left_panel)
        
        # 中间面板 - 文本编辑
        middle_panel = self.create_middle_panel()
        middle_panel.setMinimumWidth(400)
        main_splitter.addWidget(middle_panel)
        
        # 右侧面板 - 数据集分析
        right_panel = self.create_right_panel()
        right_panel.setMinimumWidth(300)
        main_splitter.addWidget(right_panel)
        
        # 设置分割器比例 (2:2:1) - 给左侧更多空间
        main_splitter.setSizes([500, 500, 300])
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setStretchFactor(2, 1)
        
        # 设置分割器样式
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #bdc3c7;
                width: 4px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #95a5a6;
            }
            QSplitter::handle:pressed {
                background-color: #7f8c8d;
            }
        """)
        
        # 设置分割器可以自由调节
        main_splitter.setChildrenCollapsible(False)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(main_splitter)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("语音库管理系统已启动")
    
    def _darken_color(self, color, factor=0.2):
        """将颜色变暗"""
        import re
        # 解析十六进制颜色
        match = re.match(r'#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})', color)
        if match:
            r, g, b = [int(x, 16) for x in match.groups()]
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            return f"#{r:02x}{g:02x}{b:02x}"
        return color
    
    def create_left_panel(self):
        """创建左侧面板 - 分类管理"""
        panel = QWidget()
        
        # 使用垂直分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 主要分类区域
        categories_widget = QWidget()
        categories_layout = QVBoxLayout(categories_widget)
        
        # 标题
        title = QLabel("📚 语音播报分类管理")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        categories_layout.addWidget(title)
        
        
        # 四种主要分类
        categories_group = QGroupBox("🎯 主要分类（点击选择）")
        categories_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 10px; padding-top: 15px; font-size: 18px; }")
        categories_group_layout = QVBoxLayout(categories_group)
        
        self.category_buttons = {}
        self.current_category = None
        self.current_subcategory = None
        self.current_text_key = None
        self.current_subsubcategory = None  # 当前子子分类
        
        # 自动记忆功能
        self.memory_file = "voice_library_memory.json"
        self.load_memory()
        categories = [
            ("🚗 障碍类型", "obstacle_types", "#e74c3c", "车辆、人员、物体、地面危险等障碍物提示"),
            ("🛣️ 用户行进路线提示", "route_guidance", "#3498db", "方向指引、路径提示、导航信息"),
            ("🔮 动态障碍物轨迹预测", "trajectory_prediction", "#f39c12", "接近预测、碰撞预警、运动分析"),
            ("🌍 环境事物提示", "environment_awareness", "#27ae60", "天气、声音、光线、交通状况")
        ]
        
        for name, key, color, desc in categories:
            # 分类按钮
            btn = QPushButton(name)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                    padding: 20px;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    min-height: 70px;
                    margin: 8px 5px;
                }}
                QPushButton:hover {{
                    background-color: {self._darken_color(color)};
                    font-size: 18px;
                }}
                QPushButton:pressed {{
                    background-color: {self._darken_color(color, 0.3)};
                }}
            """)
            btn.clicked.connect(lambda checked, k=key: self.select_category(k))
            self.category_buttons[key] = btn
            categories_group_layout.addWidget(btn)
        
        categories_layout.addWidget(categories_group)
        
        # 子分类和文本管理区域
        management_widget = QWidget()
        management_layout = QVBoxLayout(management_widget)
        
        # 子分类列表
        subcategory_group = QGroupBox("📁 子分类管理")
        subcategory_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 15px; padding-top: 15px; }")
        subcategory_layout = QVBoxLayout(subcategory_group)
        subcategory_layout.setSpacing(10)
        
        # 当前选择的分类显示
        self.current_category_label = QLabel("请先选择主要分类")
        self.current_category_label.setStyleSheet("font-size: 12px; color: #e67e22; font-weight: bold; padding: 5px; background-color: #fef9e7; border-radius: 3px;")
        subcategory_layout.addWidget(self.current_category_label)
        
        
        self.subcategory_list = QListWidget()
        self.subcategory_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #ecf0f1;
            }
        """)
        self.subcategory_list.itemClicked.connect(self.on_subcategory_selected)
        subcategory_layout.addWidget(self.subcategory_list)
        
        # 子分类管理按钮
        subcategory_btn_layout = QHBoxLayout()
        
        add_subcategory_btn = QPushButton("➕ 添加子分类")
        add_subcategory_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        add_subcategory_btn.clicked.connect(self.add_subcategory)
        subcategory_btn_layout.addWidget(add_subcategory_btn)
        
        delete_subcategory_btn = QPushButton("🗑️ 删除子分类")
        delete_subcategory_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        delete_subcategory_btn.clicked.connect(self.delete_subcategory)
        subcategory_btn_layout.addWidget(delete_subcategory_btn)
        
        subcategory_layout.addLayout(subcategory_btn_layout)
        management_layout.addWidget(subcategory_group)
        
        # 文本列表
        text_group = QGroupBox("📝 语音文本管理")
        text_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 15px; padding-top: 15px; }")
        text_layout = QVBoxLayout(text_group)
        text_layout.setSpacing(10)
        
        # 当前选择的子分类显示
        self.current_subcategory_label = QLabel("请先选择子分类")
        self.current_subcategory_label.setStyleSheet("font-size: 12px; color: #e67e22; font-weight: bold; padding: 5px; background-color: #fef9e7; border-radius: 3px;")
        text_layout.addWidget(self.current_subcategory_label)
        
        
        self.text_list = QListWidget()
        self.text_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #ecf0f1;
            }
        """)
        self.text_list.itemClicked.connect(self.on_text_selected)
        text_layout.addWidget(self.text_list)
        
        # 文本管理按钮
        text_btn_layout = QHBoxLayout()
        
        add_text_btn = QPushButton("➕ 添加文本")
        add_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        add_text_btn.clicked.connect(self.add_text)
        text_btn_layout.addWidget(add_text_btn)
        
        delete_text_btn = QPushButton("🗑️ 删除文本")
        delete_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        delete_text_btn.clicked.connect(self.delete_text)
        text_btn_layout.addWidget(delete_text_btn)
        
        text_layout.addLayout(text_btn_layout)
        management_layout.addWidget(text_group)
        
        # 添加到分割器
        splitter.addWidget(categories_widget)
        splitter.addWidget(management_widget)
        
        # 设置分割器比例 - 给主要分类区域更多空间
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        # 设置分割器样式
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #bdc3c7;
                height: 4px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #95a5a6;
            }
            QSplitter::handle:pressed {
                background-color: #7f8c8d;
            }
        """)
        
        # 设置分割器可以自由调节
        splitter.setChildrenCollapsible(False)
        
        # 主布局
        main_layout = QVBoxLayout(panel)
        main_layout.addWidget(splitter)
        
        return panel
    
    def create_middle_panel(self):
        """创建中间面板 - 文本编辑"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("✏️ 文本编辑与语音测试")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        layout.addWidget(title)
        
        # 当前编辑信息
        self.current_edit_label = QLabel("请先在左侧选择要编辑的文本")
        self.current_edit_label.setStyleSheet("font-size: 14px; color: #e67e22; font-weight: bold; margin: 5px; padding: 8px; background-color: #fef9e7; border-radius: 3px;")
        layout.addWidget(self.current_edit_label)
        
        
        # 文本编辑区域
        edit_group = QGroupBox("文本编辑")
        edit_layout = QVBoxLayout(edit_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("在此输入或编辑语音播报文本...")
        self.text_edit.setMinimumHeight(200)
        edit_layout.addWidget(self.text_edit)
        
        # 编辑按钮
        edit_btn_layout = QHBoxLayout()
        
        save_text_btn = QPushButton("保存文本")
        save_text_btn.clicked.connect(self.save_text)
        edit_btn_layout.addWidget(save_text_btn)
        
        clear_text_btn = QPushButton("清空文本")
        clear_text_btn.clicked.connect(self.clear_text)
        edit_btn_layout.addWidget(clear_text_btn)
        
        edit_layout.addLayout(edit_btn_layout)
        layout.addWidget(edit_group)
        
        # 语音测试区域
        test_group = QGroupBox("语音测试")
        test_layout = QVBoxLayout(test_group)
        
        # 语音设置
        voice_settings_layout = QHBoxLayout()
        
        voice_settings_layout.addWidget(QLabel("语速:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 300)
        self.speed_slider.setValue(150)
        voice_settings_layout.addWidget(self.speed_slider)
        
        voice_settings_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        voice_settings_layout.addWidget(self.volume_slider)
        
        test_layout.addLayout(voice_settings_layout)
        
        # 测试按钮
        test_btn_layout = QHBoxLayout()
        
        test_voice_btn = QPushButton("测试语音")
        test_voice_btn.clicked.connect(self.test_voice)
        test_btn_layout.addWidget(test_voice_btn)
        
        stop_voice_btn = QPushButton("停止语音")
        stop_voice_btn.clicked.connect(self.stop_voice)
        test_btn_layout.addWidget(stop_voice_btn)
        
        test_layout.addLayout(test_btn_layout)
        layout.addWidget(test_group)
        
        # 批量操作区域
        batch_group = QGroupBox("批量操作")
        batch_layout = QVBoxLayout(batch_group)
        
        # 批量导入
        import_layout = QHBoxLayout()
        
        self.batch_text_edit = QTextEdit()
        self.batch_text_edit.setPlaceholderText("在此粘贴批量文本，每行一个...")
        self.batch_text_edit.setMaximumHeight(100)
        import_layout.addWidget(self.batch_text_edit)
        
        import_btn = QPushButton("批量导入")
        import_btn.clicked.connect(self.batch_import)
        import_layout.addWidget(import_btn)
        
        batch_layout.addLayout(import_layout)
        layout.addWidget(batch_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """创建右侧面板 - 数据集分析"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("🔍 数据集分析与导入")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        layout.addWidget(title)
        
        
        # 手动输入数据集
        manual_group = QGroupBox("手动输入数据集")
        manual_layout = QVBoxLayout(manual_group)
        
        self.dataset_text_edit = QTextEdit()
        self.dataset_text_edit.setPlaceholderText("在此输入数据集文本，系统将自动分析并分类...")
        self.dataset_text_edit.setMaximumHeight(150)
        manual_layout.addWidget(self.dataset_text_edit)
        
        analyze_btn = QPushButton("分析数据集")
        analyze_btn.clicked.connect(self.analyze_dataset)
        manual_layout.addWidget(analyze_btn)
        
        layout.addWidget(manual_group)
        
        # 网址解析
        url_group = QGroupBox("网址解析")
        url_layout = QVBoxLayout(url_group)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("输入数据集网址...")
        url_layout.addWidget(self.url_input)
        
        url_btn_layout = QHBoxLayout()
        
        parse_url_btn = QPushButton("解析网址")
        parse_url_btn.clicked.connect(self.parse_url)
        url_btn_layout.addWidget(parse_url_btn)
        
        fetch_content_btn = QPushButton("获取内容")
        fetch_content_btn.clicked.connect(self.fetch_url_content)
        url_btn_layout.addWidget(fetch_content_btn)
        
        url_layout.addLayout(url_btn_layout)
        layout.addWidget(url_group)
        
        # 分析结果
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        # 结果操作按钮
        results_btn_layout = QHBoxLayout()
        
        apply_results_btn = QPushButton("应用结果")
        apply_results_btn.clicked.connect(self.apply_analysis_results)
        results_btn_layout.addWidget(apply_results_btn)
        
        clear_results_btn = QPushButton("清空结果")
        clear_results_btn.clicked.connect(self.clear_results)
        results_btn_layout.addWidget(clear_results_btn)
        
        results_layout.addLayout(results_btn_layout)
        layout.addWidget(results_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("暂无统计信息")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        refresh_stats_btn = QPushButton("刷新统计")
        refresh_stats_btn.clicked.connect(self.refresh_stats)
        stats_layout.addWidget(refresh_stats_btn)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return panel
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        save_action = QAction('保存语音库', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_library)
        file_menu.addAction(save_action)
        
        load_action = QAction('加载语音库', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_library)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('导出语音库', self)
        export_action.triggered.connect(self.export_library)
        file_menu.addAction(export_action)
        
        import_action = QAction('导入语音库', self)
        import_action.triggered.connect(self.import_library)
        file_menu.addAction(import_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        backup_action = QAction('备份语音库', self)
        backup_action.triggered.connect(self.backup_library)
        tools_menu.addAction(backup_action)
        
        restore_action = QAction('恢复语音库', self)
        restore_action.triggered.connect(self.restore_library)
        tools_menu.addAction(restore_action)
        
        tools_menu.addSeparator()
        
        settings_action = QAction('设置', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_library(self):
        """加载语音库"""
        try:
            if os.path.exists(self.library_file):
                with open(self.library_file, 'r', encoding='utf-8') as f:
                    self.text_library = json.load(f)
            else:
                # 创建默认语音库
                self.text_library = {
                    "obstacle_types": {
                        "static_obstacle": {
                            "name": "静态障碍",
                            "items": {
                                "vehicle": "前方有停放车辆，请绕行",
                                "trash_bin": "前方有垃圾桶，请绕行",
                                "stall": "前方有摊位，请小心通过",
                                "furniture": "前方有家具，请注意避让",
                                "barrier": "前方有栏杆，请绕行",
                                "pole": "前方有柱子，请注意避让",
                                "tree": "前方有树木，请小心通过",
                                "sign": "前方有标志牌，请注意"
                            }
                        },
                        "dynamic_obstacle": {
                            "name": "动态障碍",
                            "items": {
                                "pedestrian": "前方有行人，请减速慢行",
                                "child": "前方有儿童，请特别小心",
                                "elderly": "前方有老人，请减速让行",
                                "wheelchair": "前方有轮椅使用者，请让行",
                                "group": "前方有多人聚集，请小心通过",
                                "moving_vehicle": "前方有移动车辆，请注意避让",
                                "pet": "前方有宠物，请小心通过"
                            }
                        },
                        "ground_anomaly": {
                            "name": "地面异常",
                            "items": {
                                "pothole": "前方有坑洼，请绕行",
                                "water": "前方有积水，请小心通过",
                                "ice": "前方有冰面，请特别小心",
                                "step": "前方有台阶，请小心上下",
                                "slope": "前方有斜坡，请减速通过",
                                "construction": "前方有施工区域，请绕行",
                                "crack": "前方有地面裂缝，请小心通过"
                            }
                        }
                    },
                    "route_guidance": {
                        "direction": {
                            "name": "方向指引",
                            "items": {
                                "straight": "请直行",
                                "turn_left": "请左转",
                                "turn_right": "请右转",
                                "u_turn": "请掉头"
                            }
                        },
                        "path": {
                            "name": "路径提示",
                            "items": {
                                "blind_path": "盲道清晰，请沿盲道前进",
                                "path_interrupted": "盲道中断，请寻找其他路径",
                                "narrow_path": "通道狭窄，请减速通过"
                            }
                        }
                    },
                    "trajectory_prediction": {
                        "approaching": {
                            "name": "接近预测",
                            "items": {
                                "person_approaching": "有行人正在接近，请等待",
                                "vehicle_approaching": "有车辆正在接近，请避让",
                                "object_approaching": "有物体正在接近，请注意"
                            }
                        },
                        "collision": {
                            "name": "碰撞预警",
                            "items": {
                                "collision_warning": "预测可能发生碰撞，请立即停止",
                                "path_crossing": "预测路径交叉，请谨慎通过",
                                "safe_passage": "预测安全，可以正常通过"
                            }
                        }
                    },
                    "environment_awareness": {
                        "weather": {
                            "name": "天气状况",
                            "items": {
                                "sunny": "天气晴朗，环境良好，可以正常出行",
                                "rainy": "正在下雨，路面湿滑，请小心行走",
                                "foggy": "有雾，能见度较低，请减速慢行",
                                "night": "夜晚光线较暗，请注意安全"
                            }
                        },
                        "sound": {
                            "name": "声音提示",
                            "items": {
                                "car_horn": "前方有汽车鸣笛，请注意安全",
                                "construction": "前方有施工噪音，请减速通过",
                                "alarm": "前方有警报声，请提高警惕"
                            }
                        }
                    }
                }
                self.save_library()
            
            self.refresh_category_buttons()
            self.statusBar().showMessage("语音库加载成功")
            return True
            
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"加载语音库失败: {e}")
            self.statusBar().showMessage("语音库加载失败")
            return False
    
    def save_library(self):
        """保存语音库"""
        try:
            with open(self.library_file, 'w', encoding='utf-8') as f:
                json.dump(self.text_library, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage("语音库保存成功")
            return True
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存语音库失败: {e}")
            self.statusBar().showMessage("语音库保存失败")
            return False
    
    def refresh_category_buttons(self):
        """刷新分类按钮状态"""
        for key, btn in self.category_buttons.items():
            if key in self.text_library:
                btn.setStyleSheet(btn.styleSheet() + " border: 2px solid #2c3e50;")
            else:
                btn.setStyleSheet(btn.styleSheet().replace(" border: 2px solid #2c3e50;", ""))
    
    def select_category(self, category_key):
        """选择分类"""
        self.current_category = category_key
        
        # 更新按钮状态
        for key, btn in self.category_buttons.items():
            if key == category_key:
                # 选中状态 - 添加更明显的边框和阴影
                current_style = btn.styleSheet()
                if "border: 4px solid #2c3e50;" not in current_style:
                    btn.setStyleSheet(current_style + " border: 4px solid #2c3e50; box-shadow: 0 4px 8px rgba(0,0,0,0.3);")
            else:
                # 未选中状态 - 移除选中样式
                current_style = btn.styleSheet()
                btn.setStyleSheet(current_style.replace(" border: 4px solid #2c3e50; box-shadow: 0 4px 8px rgba(0,0,0,0.3);", ""))
        
        # 更新当前分类标签
        category_names = {
            "obstacle_types": "障碍类型",
            "route_guidance": "用户行进路线提示", 
            "trajectory_prediction": "动态障碍物轨迹预测",
            "environment_awareness": "环境事物提示"
        }
        
        self.current_category_label.setText(f"当前分类: {category_names.get(category_key, category_key)}")
        self.current_category_label.setStyleSheet("font-size: 12px; color: #27ae60; font-weight: bold; padding: 5px; background-color: #d5f4e6; border-radius: 3px;")
        
        self.refresh_subcategories()
        self.save_memory()  # 保存记忆状态
        self.statusBar().showMessage(f"已选择分类: {category_names.get(category_key, category_key)}")
    
    def refresh_subcategories(self):
        """刷新子分类列表"""
        self.subcategory_list.clear()
        if self.current_category in self.text_library:
            for sub_key, sub_data in self.text_library[self.current_category].items():
                # 检查是否有子子分类
                has_subsub = 'subcategories' in sub_data and sub_data['subcategories']
                display_name = sub_data.get('name', sub_key)
                if has_subsub:
                    display_name += f" ({len(sub_data['subcategories'])}个子类)"
                
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, sub_key)
                self.subcategory_list.addItem(item)
    
    def on_subcategory_selected(self, item):
        """选择子分类"""
        try:
            if item:
                self.current_subcategory = item.data(Qt.UserRole)
                
                # 检查是否有子子分类
                if (self.current_category in self.text_library and 
                    self.current_subcategory in self.text_library[self.current_category]):
                    sub_data = self.text_library[self.current_category][self.current_subcategory]
                    has_subsub = 'subcategories' in sub_data and sub_data['subcategories']
                    
                    if has_subsub:
                        # 如果有子子分类，显示子子分类列表
                        self.current_subcategory_label.setText(f"当前子分类: {sub_data.get('name', self.current_subcategory)} (包含子类)")
                        self.current_subcategory_label.setStyleSheet("font-size: 12px; color: #e67e22; font-weight: bold; padding: 5px; background-color: #fef9e7; border-radius: 3px;")
                        self.refresh_subsubcategories()
                    else:
                        # 如果没有子子分类，显示文本列表
                        self.current_subcategory_label.setText(f"当前子分类: {sub_data.get('name', self.current_subcategory)}")
                        self.current_subcategory_label.setStyleSheet("font-size: 12px; color: #27ae60; font-weight: bold; padding: 5px; background-color: #d5f4e6; border-radius: 3px;")
                        self.refresh_texts()
                    
                    self.save_memory()  # 保存记忆状态
                    self.statusBar().showMessage(f"已选择子分类: {item.text()}")
        except Exception as e:
            print(f"选择子分类时出错: {e}")
            QMessageBox.warning(self, "错误", f"选择子分类时出错: {e}")
    
    def load_memory(self):
        """加载记忆状态"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    self.current_category = memory.get('current_category')
                    self.current_subcategory = memory.get('current_subcategory')
                    self.current_subsubcategory = memory.get('current_subsubcategory')
        except Exception as e:
            print(f"加载记忆失败: {e}")
    
    def save_memory(self):
        """保存记忆状态"""
        try:
            memory = {
                'current_category': self.current_category,
                'current_subcategory': self.current_subcategory,
                'current_subsubcategory': self.current_subsubcategory
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")
    
    def apply_memory(self):
        """应用记忆状态"""
        try:
            # 延迟应用记忆状态，确保界面完全初始化
            QTimer.singleShot(100, self._apply_memory_delayed)
        except Exception as e:
            print(f"应用记忆状态时出错: {e}")
            # 如果记忆状态有问题，重置为默认状态
            self.current_category = None
            self.current_subcategory = None
            self.current_subsubcategory = None
    
    def _apply_memory_delayed(self):
        """延迟应用记忆状态"""
        try:
            if self.current_category and self.current_category in self.text_library:
                # 恢复主要分类选择
                if self.current_category in self.category_buttons:
                    self.select_category(self.current_category)
                
                if self.current_subcategory and self.current_subcategory in self.text_library[self.current_category]:
                    # 恢复子分类选择
                    self.refresh_subcategories()
                    # 查找并选择对应的子分类
                    for i in range(self.subcategory_list.count()):
                        item = self.subcategory_list.item(i)
                        if item.data(Qt.UserRole) == self.current_subcategory:
                            self.subcategory_list.setCurrentItem(item)
                            self.on_subcategory_selected(item)
                            break
                    
                    if (self.current_subsubcategory and 
                        'subcategories' in self.text_library[self.current_category][self.current_subcategory] and
                        self.current_subsubcategory in self.text_library[self.current_category][self.current_subcategory]['subcategories']):
                        # 恢复子子分类选择
                        self.refresh_subsubcategories()
                        for i in range(self.text_list.count()):
                            item = self.text_list.item(i)
                            if item.data(Qt.UserRole) == self.current_subsubcategory:
                                self.text_list.setCurrentItem(item)
                                self.on_text_selected(item)
                                break
        except Exception as e:
            print(f"延迟应用记忆状态时出错: {e}")
            # 如果记忆状态有问题，重置为默认状态
            self.current_category = None
            self.current_subcategory = None
            self.current_subsubcategory = None
    
    def refresh_subsubcategories(self):
        """刷新子子分类列表"""
        self.text_list.clear()
        if (self.current_category in self.text_library and 
            self.current_subcategory in self.text_library[self.current_category]):
            sub_data = self.text_library[self.current_category][self.current_subcategory]
            if 'subcategories' in sub_data:
                for subsub_key, subsub_data in sub_data['subcategories'].items():
                    item = QListWidgetItem(subsub_data.get('name', subsub_key))
                    item.setData(Qt.UserRole, subsub_key)
                    self.text_list.addItem(item)
    
    def refresh_texts(self):
        """刷新文本列表"""
        self.text_list.clear()
        if (self.current_category in self.text_library and 
            self.current_subcategory in self.text_library[self.current_category]):
            items = self.text_library[self.current_category][self.current_subcategory]['items']
            for key, text in items.items():
                item = QListWidgetItem(f"{key}: {text}")
                item.setData(Qt.UserRole, key)
                self.text_list.addItem(item)
    
    def on_text_selected(self, item):
        """选择文本或子子分类"""
        try:
            if item:
                self.current_text_key = item.data(Qt.UserRole)
                
                # 检查当前是否在子子分类模式
                if (self.current_category in self.text_library and 
                    self.current_subcategory in self.text_library[self.current_category]):
                    sub_data = self.text_library[self.current_category][self.current_subcategory]
                    has_subsub = 'subcategories' in sub_data and sub_data['subcategories']
                    
                    if has_subsub:
                        # 子子分类模式，检查选中的是否是子子分类
                        if self.current_text_key in sub_data['subcategories']:
                            subsub_data = sub_data['subcategories'][self.current_text_key]
                            self.current_subsubcategory = self.current_text_key
                            
                            # 检查子子分类是否还有更深层的子分类
                            if 'subcategories' in subsub_data and subsub_data['subcategories']:
                                # 有更深层的子分类，显示子子子分类列表
                                self.text_list.clear()
                                for subsubsub_key, subsubsub_data in subsub_data['subcategories'].items():
                                    has_subsubsub = 'subcategories' in subsubsub_data and subsubsub_data['subcategories']
                                    display_name = subsubsub_data.get('name', subsubsub_key)
                                    if has_subsubsub:
                                        display_name += f" ({len(subsubsub_data['subcategories'])}个子类)"
                                    
                                    text_item = QListWidgetItem(display_name)
                                    text_item.setData(Qt.UserRole, subsubsub_key)
                                    self.text_list.addItem(text_item)
                                self.current_edit_label.setText(f"正在查看子子分类: {subsub_data.get('name', self.current_text_key)}")
                            elif 'items' in subsub_data:
                                # 显示子子分类的文本列表
                                self.text_list.clear()
                                for text_key, text_value in subsub_data['items'].items():
                                    text_item = QListWidgetItem(f"{text_key}: {text_value}")
                                    text_item.setData(Qt.UserRole, text_key)
                                    self.text_list.addItem(text_item)
                                self.current_edit_label.setText(f"正在查看子子分类: {subsub_data.get('name', self.current_text_key)}")
                            else:
                                # 空的子子分类，可以添加内容
                                self.text_list.clear()
                                self.current_edit_label.setText(f"正在编辑子子分类: {subsub_data.get('name', self.current_text_key)}")
                    else:
                        # 普通文本模式
                        text = item.text().split(": ", 1)[1]
                        self.text_edit.setText(text)
                        self.current_edit_label.setText(f"正在编辑: {self.current_text_key}")
                    
                    self.save_memory()  # 保存记忆状态
        except Exception as e:
            print(f"选择文本时出错: {e}")
            QMessageBox.warning(self, "错误", f"选择文本时出错: {e}")
    
    def add_subcategory(self):
        """添加子分类"""
        if not hasattr(self, 'current_category') or not self.current_category:
            QMessageBox.warning(self, "提示", "请先选择主要分类")
            return
        
        # 确定添加位置：主要分类下还是子子分类下
        target_category = self.current_category
        target_subcategory = self.current_subcategory
        target_subsubcategory = self.current_subsubcategory
        
        # 确保目标分类存在，如果不存在则创建
        if target_category not in self.text_library:
            self.text_library[target_category] = {}
        
        # 如果当前在子子分类模式且有选中的子子分类，则在子子分类下添加
        if (target_subcategory and 
            target_subcategory in self.text_library[target_category] and
            'subcategories' in self.text_library[target_category][target_subcategory] and
            target_subsubcategory and
            target_subsubcategory in self.text_library[target_category][target_subcategory]['subcategories']):
            # 在子子分类下添加子子子分类
            parent_data = self.text_library[target_category][target_subcategory]['subcategories'][target_subsubcategory]
            parent_name = parent_data.get('name', target_subsubcategory)
        elif (target_subcategory and 
              target_subcategory in self.text_library[target_category] and
              'subcategories' in self.text_library[target_category][target_subcategory]):
            # 在子分类下添加子子分类
            parent_data = self.text_library[target_category][target_subcategory]
            parent_name = parent_data.get('name', target_subcategory)
        else:
            # 在主要分类下添加子分类
            parent_data = self.text_library[target_category]
            parent_name = target_category
        
        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("添加子分类")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 说明文字
        info_label = QLabel("📝 添加新的子分类")
        info_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(info_label)
        
        # 当前分类显示
        current_cat_label = QLabel(f"当前父级分类: {parent_name}")
        current_cat_label.setStyleSheet("font-size: 14px; color: #7f8c8d; margin: 5px;")
        layout.addWidget(current_cat_label)
        
        # 子分类类型选择
        type_label = QLabel("子分类类型:")
        type_label.setStyleSheet("font-weight: bold; margin-top: 10px; font-size: 16px;")
        layout.addWidget(type_label)
        
        type_combo = QComboBox()
        type_combo.addItems(["普通子分类", "包含子类的子分类"])
        type_combo.setStyleSheet("padding: 8px; border: 2px solid #bdc3c7; border-radius: 4px; font-size: 16px;")
        layout.addWidget(type_combo)
        
        # 子分类名称输入
        name_label = QLabel("子分类名称:")
        name_label.setStyleSheet("font-weight: bold; margin-top: 10px; font-size: 16px;")
        layout.addWidget(name_label)
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("例如: 车辆类、人员类、物体类等")
        name_input.setStyleSheet("padding: 10px; border: 2px solid #bdc3c7; border-radius: 4px; font-size: 16px;")
        layout.addWidget(name_input)
        
        # 说明
        info_text = QLabel("💡 系统会自动生成内部标识，无需手动输入")
        info_text.setStyleSheet("font-size: 14px; color: #7f8c8d; margin: 5px; padding: 8px; background-color: #f8f9fa; border-radius: 3px;")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # 连接信号
        def on_ok():
            name = name_input.text().strip()
            subcategory_type = type_combo.currentText()
            
            if not name:
                QMessageBox.warning(dialog, "错误", "请输入子分类名称")
                return
            
            # 自动生成标识
            key = self.generate_subcategory_key(name)
            
            # 检查标识是否已存在并添加子分类
            if target_subsubcategory and target_subsubcategory in parent_data:
                # 在子子分类下添加子子子分类
                if 'subcategories' not in parent_data:
                    parent_data['subcategories'] = {}
                if key in parent_data['subcategories']:
                    counter = 1
                    original_key = key
                    while key in parent_data['subcategories']:
                        key = f"{original_key}_{counter}"
                        counter += 1
                
                if subcategory_type == "包含子类的子分类":
                    parent_data['subcategories'][key] = {
                        "name": name,
                        "subcategories": {}
                    }
                else:
                    parent_data['subcategories'][key] = {
                        "name": name,
                        "items": {}
                    }
            elif target_subcategory and target_subcategory in self.text_library[target_category]:
                # 在子分类下添加子子分类
                if 'subcategories' not in self.text_library[target_category][target_subcategory]:
                    self.text_library[target_category][target_subcategory]['subcategories'] = {}
                if key in self.text_library[target_category][target_subcategory]['subcategories']:
                    counter = 1
                    original_key = key
                    while key in self.text_library[target_category][target_subcategory]['subcategories']:
                        key = f"{original_key}_{counter}"
                        counter += 1
                
                if subcategory_type == "包含子类的子分类":
                    self.text_library[target_category][target_subcategory]['subcategories'][key] = {
                        "name": name,
                        "subcategories": {}
                    }
                else:
                    self.text_library[target_category][target_subcategory]['subcategories'][key] = {
                        "name": name,
                        "items": {}
                    }
            else:
                # 在主要分类下添加子分类
                if target_category not in self.text_library:
                    self.text_library[target_category] = {}
                if key in self.text_library[target_category]:
                    counter = 1
                    original_key = key
                    while key in self.text_library[target_category]:
                        key = f"{original_key}_{counter}"
                        counter += 1
                
                if subcategory_type == "包含子类的子分类":
                    self.text_library[target_category][key] = {
                        "name": name,
                        "subcategories": {}
                    }
                else:
                    self.text_library[target_category][key] = {
                        "name": name,
                        "items": {}
                    }
            
            self.refresh_subcategories()
            dialog.accept()
            QMessageBox.information(self, "添加成功", f"子分类 '{name}' 已添加")
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    def generate_subcategory_key(self, name):
        """根据名称自动生成子分类标识"""
        import re
        # 移除特殊字符，只保留中文、英文、数字
        key = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', name)
        
        # 如果是中文，转换为拼音（简化版）
        if re.search(r'[\u4e00-\u9fa5]', key):
            # 简单的中文转拼音映射
            chinese_map = {
                '车辆': 'vehicle', '汽车': 'car', '卡车': 'truck', '摩托车': 'motorcycle', '自行车': 'bicycle',
                '人员': 'person', '行人': 'pedestrian', '儿童': 'child', '老人': 'elderly', '轮椅': 'wheelchair',
                '物体': 'object', '垃圾桶': 'trash_bin', '摊位': 'stall', '家具': 'furniture', '栏杆': 'barrier',
                '地面': 'ground', '坑洼': 'pothole', '积水': 'water', '台阶': 'step', '斜坡': 'slope',
                '方向': 'direction', '直行': 'straight', '左转': 'turn_left', '右转': 'turn_right',
                '路径': 'path', '盲道': 'blind_path', '导航': 'navigation',
                '预测': 'prediction', '接近': 'approaching', '碰撞': 'collision', '运动': 'movement',
                '天气': 'weather', '晴天': 'sunny', '雨天': 'rainy', '雾天': 'foggy', '夜晚': 'night',
                '声音': 'sound', '鸣笛': 'horn', '施工': 'construction', '警报': 'alarm',
                '光线': 'lighting', '明亮': 'bright', '昏暗': 'dim', '黑暗': 'dark',
                '交通': 'traffic', '繁忙': 'busy', '清静': 'quiet', '拥堵': 'congested'
            }
            
            # 查找匹配的中文词汇
            for chinese, english in chinese_map.items():
                if chinese in key:
                    key = key.replace(chinese, english)
                    break
            
            # 如果还有中文字符，使用通用标识
            if re.search(r'[\u4e00-\u9fa5]', key):
                key = 'category'
        
        # 转换为小写，替换空格为下划线
        key = key.lower().replace(' ', '_')
        
        # 如果为空或太短，使用默认值
        if len(key) < 2:
            key = 'category'
        
        return key
    
    def delete_subcategory(self):
        """删除子分类"""
        current_item = self.subcategory_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "请先选择要删除的子分类")
            return
        
        reply = QMessageBox.question(self, "确认删除", 
                                   f"确定要删除子分类 '{current_item.text()}' 吗？")
        if reply == QMessageBox.Yes:
            key = current_item.data(Qt.UserRole)
            del self.text_library[self.current_category][key]
            self.refresh_subcategories()
            QMessageBox.information(self, "删除成功", "子分类已删除")
    
    def add_text(self):
        """添加文本"""
        if not hasattr(self, 'current_subcategory'):
            QMessageBox.warning(self, "提示", "请先选择子分类")
            return
        
        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("添加语音文本")
        dialog.setModal(True)
        dialog.resize(400, 250)
        
        layout = QVBoxLayout(dialog)
        
        # 说明文字
        info_label = QLabel("📝 添加新的语音文本")
        info_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(info_label)
        
        # 当前子分类显示
        current_subcat_label = QLabel(f"当前子分类: {self.current_subcategory}")
        current_subcat_label.setStyleSheet("font-size: 14px; color: #7f8c8d; margin: 5px;")
        layout.addWidget(current_subcat_label)
        
        # 文本内容输入
        text_label = QLabel("语音文本内容:")
        text_label.setStyleSheet("font-weight: bold; margin-top: 10px; font-size: 16px;")
        layout.addWidget(text_label)
        
        text_input = QTextEdit()
        text_input.setPlaceholderText("例如: 前方有汽车，请注意避让")
        text_input.setStyleSheet("padding: 10px; border: 2px solid #bdc3c7; border-radius: 4px; font-size: 16px; min-height: 80px;")
        layout.addWidget(text_input)
        
        # 说明
        info_text = QLabel("💡 系统会自动生成文本标识，无需手动输入")
        info_text.setStyleSheet("font-size: 14px; color: #7f8c8d; margin: 5px; padding: 8px; background-color: #f8f9fa; border-radius: 3px;")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # 连接信号
        def on_ok():
            text = text_input.toPlainText().strip()
            
            if not text:
                QMessageBox.warning(dialog, "错误", "请输入语音文本内容")
                return
            
            # 自动生成标识
            key = self.generate_text_key(text)
            
            # 检查标识是否已存在
            if key in self.text_library[self.current_category][self.current_subcategory]['items']:
                # 如果存在，添加数字后缀
                counter = 1
                original_key = key
                while key in self.text_library[self.current_category][self.current_subcategory]['items']:
                    key = f"{original_key}_{counter}"
                    counter += 1
            
            # 添加文本
            self.text_library[self.current_category][self.current_subcategory]['items'][key] = text
            
            self.refresh_texts()
            dialog.accept()
            QMessageBox.information(self, "添加成功", f"语音文本已添加")
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    def generate_text_key(self, text):
        """根据文本内容自动生成文本标识"""
        import re
        # 提取关键词
        keywords = []
        
        # 常见关键词映射
        keyword_map = {
            '汽车': 'car', '车辆': 'vehicle', '卡车': 'truck', '摩托车': 'motorcycle', '自行车': 'bicycle',
            '行人': 'pedestrian', '人员': 'person', '儿童': 'child', '老人': 'elderly', '轮椅': 'wheelchair',
            '垃圾桶': 'trash_bin', '摊位': 'stall', '家具': 'furniture', '栏杆': 'barrier', '柱子': 'pole',
            '坑洼': 'pothole', '积水': 'water', '台阶': 'step', '斜坡': 'slope', '施工': 'construction',
            '直行': 'straight', '左转': 'turn_left', '右转': 'turn_right', '掉头': 'u_turn',
            '盲道': 'blind_path', '路径': 'path', '导航': 'navigation', '方向': 'direction',
            '接近': 'approaching', '碰撞': 'collision', '预测': 'prediction', '运动': 'movement',
            '晴天': 'sunny', '雨天': 'rainy', '雾天': 'foggy', '夜晚': 'night', '天气': 'weather',
            '鸣笛': 'horn', '施工': 'construction', '警报': 'alarm', '声音': 'sound',
            '明亮': 'bright', '昏暗': 'dim', '黑暗': 'dark', '光线': 'lighting',
            '繁忙': 'busy', '清静': 'quiet', '拥堵': 'congested', '交通': 'traffic'
        }
        
        # 查找匹配的关键词
        for chinese, english in keyword_map.items():
            if chinese in text:
                keywords.append(english)
        
        # 如果没有找到关键词，使用文本前几个字符
        if not keywords:
            # 移除标点符号和空格
            clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
            if len(clean_text) >= 2:
                keywords.append(clean_text[:4].lower())
            else:
                keywords.append('text')
        
        # 生成标识
        key = '_'.join(keywords[:2])  # 最多使用2个关键词
        
        # 如果为空或太短，使用默认值
        if len(key) < 2:
            key = 'voice_text'
        
        return key
    
    def delete_text(self):
        """删除文本"""
        current_item = self.text_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "请先选择要删除的文本")
            return
        
        key = current_item.data(Qt.UserRole)
        reply = QMessageBox.question(self, "确认删除", f"确定要删除文本 '{key}' 吗？")
        if reply == QMessageBox.Yes:
            del self.text_library[self.current_category][self.current_subcategory]['items'][key]
            self.refresh_texts()
            QMessageBox.information(self, "删除成功", "文本已删除")
    
    def save_text(self):
        """保存文本"""
        if not hasattr(self, 'current_text_key'):
            QMessageBox.warning(self, "提示", "请先选择要编辑的文本")
            return
        
        new_text = self.text_edit.toPlainText().strip()
        if new_text:
            self.text_library[self.current_category][self.current_subcategory]['items'][self.current_text_key] = new_text
            self.refresh_texts()
            QMessageBox.information(self, "保存成功", "文本已保存")
        else:
            QMessageBox.warning(self, "保存失败", "文本内容不能为空")
    
    def clear_text(self):
        """清空文本"""
        self.text_edit.clear()
        self.current_edit_label.setText("请选择要编辑的文本")
    
    def test_voice(self):
        """测试语音"""
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入要测试的文本")
            return
        
        try:
            if self.tts_engine is None:
                self.tts_engine = pyttsx3.init()
            
            # 设置语音参数
            self.tts_engine.setProperty('rate', self.speed_slider.value())
            self.tts_engine.setProperty('volume', self.volume_slider.value() / 100.0)
            
            # 在单独线程中播放语音
            def play_voice():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=play_voice)
            thread.daemon = True
            thread.start()
            
            self.statusBar().showMessage("正在播放语音...")
            
        except Exception as e:
            QMessageBox.warning(self, "语音测试失败", f"语音测试失败: {e}")
    
    def stop_voice(self):
        """停止语音"""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
            self.statusBar().showMessage("语音已停止")
        except Exception as e:
            QMessageBox.warning(self, "停止失败", f"停止语音失败: {e}")
    
    def batch_import(self):
        """批量导入"""
        text = self.batch_text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入要导入的文本")
            return
        
        if not hasattr(self, 'current_subcategory'):
            QMessageBox.warning(self, "提示", "请先选择子分类")
            return
        
        lines = text.split('\n')
        imported_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                key = f"imported_{i+1}"
                self.text_library[self.current_category][self.current_subcategory]['items'][key] = line
                imported_count += 1
        
        self.refresh_texts()
        QMessageBox.information(self, "导入成功", f"成功导入 {imported_count} 条文本")
        self.batch_text_edit.clear()
    
    def analyze_dataset(self):
        """分析数据集"""
        text = self.dataset_text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入要分析的数据集文本")
            return
        
        try:
            results = self.dataset_analyzer.analyze_text_dataset(text)
            self.display_analysis_results(results)
            self.statusBar().showMessage("数据集分析完成")
        except Exception as e:
            QMessageBox.warning(self, "分析失败", f"数据集分析失败: {e}")
    
    def display_analysis_results(self, results):
        """显示分析结果"""
        result_text = "=== 数据集分析结果 ===\n\n"
        
        for category, items in results.items():
            if items:
                result_text += f"【{category}】\n"
                for item in items:
                    result_text += f"  - {item['text']}\n"
                result_text += "\n"
        
        self.results_text.setText(result_text)
        self.analysis_results = results
    
    def parse_url(self):
        """解析网址"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入要解析的网址")
            return
        
        try:
            result = self.url_parser.parse_url(url)
            if result.get('error'):
                QMessageBox.warning(self, "解析失败", f"网址解析失败: {result['error']}")
            else:
                info = f"网址: {result['url']}\n"
                info += f"域名: {result['domain']}\n"
                info += f"支持: {'是' if result['is_supported'] else '否'}\n"
                info += f"文件类型: {result['file_type']}\n"
                if result.get('content'):
                    info += f"内容长度: {len(result['content'])} 字符"
                self.results_text.setText(info)
                self.statusBar().showMessage("网址解析完成")
        except Exception as e:
            QMessageBox.warning(self, "解析失败", f"网址解析失败: {e}")
    
    def fetch_url_content(self):
        """获取网址内容"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入要获取内容的网址")
            return
        
        try:
            result = self.url_parser.parse_url(url)
            if result.get('content'):
                self.dataset_text_edit.setText(result['content'])
                self.statusBar().showMessage("内容获取成功")
            else:
                QMessageBox.warning(self, "获取失败", "无法获取网址内容")
        except Exception as e:
            QMessageBox.warning(self, "获取失败", f"获取内容失败: {e}")
    
    def apply_analysis_results(self):
        """应用分析结果"""
        if not self.analysis_results:
            QMessageBox.warning(self, "提示", "没有可应用的分析结果")
            return
        
        try:
            applied_count = 0
            for category, items in self.analysis_results.items():
                if items and category in self.text_library:
                    # 找到对应的子分类
                    for sub_key, sub_data in self.text_library[category].items():
                        for item in items:
                            key = f"analyzed_{applied_count + 1}"
                            sub_data['items'][key] = item['text']
                            applied_count += 1
            
            self.refresh_texts()
            QMessageBox.information(self, "应用成功", f"成功应用 {applied_count} 条分析结果")
            self.clear_results()
        except Exception as e:
            QMessageBox.warning(self, "应用失败", f"应用分析结果失败: {e}")
    
    def clear_results(self):
        """清空结果"""
        self.results_text.clear()
        self.analysis_results = {}
        self.statusBar().showMessage("结果已清空")
    
    def refresh_stats(self):
        """刷新统计信息"""
        try:
            total_categories = len(self.text_library)
            total_subcategories = sum(len(cat) for cat in self.text_library.values())
            total_texts = sum(len(sub['items']) for cat in self.text_library.values() 
                            for sub in cat.values() if isinstance(sub, dict) and 'items' in sub)
            
            stats = f"总分类数: {total_categories}\n"
            stats += f"总子分类数: {total_subcategories}\n"
            stats += f"总文本数: {total_texts}\n\n"
            
            for category, subcategories in self.text_library.items():
                stats += f"{category}: {len(subcategories)} 个子分类\n"
            
            self.stats_label.setText(stats)
            self.statusBar().showMessage("统计信息已刷新")
        except Exception as e:
            QMessageBox.warning(self, "刷新失败", f"刷新统计信息失败: {e}")
    
    def export_library(self):
        """导出语音库"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出语音库", 
                                                 "voice_library_export.json", 
                                                 "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.text_library, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "导出成功", f"语音库已导出到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"导出失败: {e}")
    
    def import_library(self):
        """导入语音库"""
        file_path, _ = QFileDialog.getOpenFileName(self, "导入语音库", 
                                                 "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text_library = json.load(f)
                self.refresh_category_buttons()
                QMessageBox.information(self, "导入成功", "语音库已导入")
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"导入失败: {e}")
    
    def backup_library(self):
        """备份语音库"""
        backup_file = f"voice_library_backup_{int(time.time())}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.text_library, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "备份成功", f"语音库已备份到: {backup_file}")
        except Exception as e:
            QMessageBox.warning(self, "备份失败", f"备份失败: {e}")
    
    def restore_library(self):
        """恢复语音库"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择备份文件", 
                                                 "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text_library = json.load(f)
                self.refresh_category_buttons()
                QMessageBox.information(self, "恢复成功", "语音库已恢复")
            except Exception as e:
                QMessageBox.warning(self, "恢复失败", f"恢复失败: {e}")
    
    def show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "设置", "设置功能待实现")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h3>语音库管理系统</h3>
        <p>版本: 1.0.0</p>
        <p>用于盲道检测项目的语音播报文本库管理</p>
        <p>支持四种文本提示类型：</p>
        <ul>
        <li>障碍类型</li>
        <li>用户行进路线提示</li>
        <li>动态障碍物轨迹预测提示</li>
        <li>环境事物提示</li>
        </ul>
        <p>支持数据集分析和网址解析功能</p>
        """
        QMessageBox.about(self, "关于", about_text)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("语音库管理系统")
    app.setApplicationVersion("1.0.0")
    
    # 设置应用图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.ico"))
    
    window = VoiceLibraryManagerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
