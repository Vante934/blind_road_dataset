#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试UI组件
提供模型测试的用户界面，包括模型选择、数据集选择、实时图表等
"""

import sys
import os
from datetime import datetime
from typing import Dict, List

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QProgressBar, QTextEdit,
                             QGroupBox, QTableWidget, QTableWidgetItem, 
                             QSplitter, QFrame, QGridLayout, QSpinBox,
                             QCheckBox, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
try:
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
    CHART_AVAILABLE = True
except ImportError:
    try:
        from PyQt5.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
        CHART_AVAILABLE = True
    except ImportError:
        CHART_AVAILABLE = False
        print("⚠️ PyQt5.QtChart 不可用，图表功能将被禁用")

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modules.model_tester import ModelTester


class ModelTestUI(QWidget):
    """模型测试UI主类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tester = ModelTester()
        self.current_session_id = None
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts)
        
        self.init_ui()
        self.connect_signals()
        self.load_available_options()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧结果面板
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割器比例
        main_splitter.setSizes([300, 700])
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 测试配置组
        config_group = QGroupBox("测试配置")
        config_layout = QVBoxLayout(config_group)
        
        # 数据集选择
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("数据集:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(150)
        dataset_layout.addWidget(self.dataset_combo)
        config_layout.addLayout(dataset_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)
        
        # 会话名称
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("会话名:"))
        self.session_edit = QLineEdit()
        self.session_edit.setPlaceholderText("自动生成")
        session_layout.addWidget(self.session_edit)
        config_layout.addLayout(session_layout)
        
        layout.addWidget(config_group)
        
        # 测试控制组
        control_group = QGroupBox("测试控制")
        control_layout = QVBoxLayout(control_group)
        
        # 开始测试按钮
        self.start_button = QPushButton("开始测试")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
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
        control_layout.addWidget(self.start_button)
        
        # 停止测试按钮
        self.stop_button = QPushButton("停止测试")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.stop_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        layout.addWidget(control_group)
        
        # 模型信息组
        info_group = QGroupBox("模型信息")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setMaximumHeight(150)
        self.model_info_text.setReadOnly(True)
        info_layout.addWidget(self.model_info_text)
        
        layout.addWidget(info_group)
        
        # 历史会话组
        history_group = QGroupBox("历史会话")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setMaximumHeight(200)
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["会话名", "模型", "数据集", "状态"])
        history_layout.addWidget(self.history_table)
        
        layout.addWidget(history_group)
        
        return panel
    
    def create_results_panel(self):
        """创建结果面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建结果分割器
        results_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(results_splitter)
        
        # 实时指标面板
        metrics_panel = self.create_metrics_panel()
        results_splitter.addWidget(metrics_panel)
        
        # 图表面板
        charts_panel = self.create_charts_panel()
        results_splitter.addWidget(charts_panel)
        
        # 设置分割器比例
        results_splitter.setSizes([300, 400])
        
        return panel
    
    def create_metrics_panel(self):
        """创建指标面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 实时指标组
        metrics_group = QGroupBox("实时指标")
        metrics_layout = QGridLayout(metrics_group)
        
        # 创建指标标签
        self.metrics_labels = {}
        metrics = [
            ("准确率", "accuracy", "0.00%"),
            ("精确率", "precision", "0.00%"),
            ("召回率", "recall", "0.00%"),
            ("F1分数", "f1_score", "0.00"),
            ("推理时间", "inference_time", "0.00 ms"),
            ("FPS", "fps", "0.00"),
            ("内存使用", "memory_usage", "0.00 MB"),
            ("检测数量", "detection_count", "0")
        ]
        
        for i, (name, key, default) in enumerate(metrics):
            label = QLabel(f"{name}:")
            value_label = QLabel(default)
            value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            row = i // 2
            col = (i % 2) * 2
            
            metrics_layout.addWidget(label, row, col)
            metrics_layout.addWidget(value_label, row, col + 1)
            
            self.metrics_labels[key] = value_label
        
        layout.addWidget(metrics_group)
        
        # 测试结果组
        results_group = QGroupBox("测试结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        return panel
    
    def create_charts_panel(self):
        """创建图表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 图表组
        charts_group = QGroupBox("性能图表")
        charts_layout = QVBoxLayout(charts_group)
        
        # 创建图表视图
        self.chart_view = QChartView()
        self.chart_view.setMinimumHeight(300)
        charts_layout.addWidget(self.chart_view)
        
        # 初始化图表
        self.init_charts()
        
        layout.addWidget(charts_group)
        
        return panel
    
    def init_charts(self):
        """初始化图表"""
        if not CHART_AVAILABLE:
            # 如果图表不可用，显示替代文本
            self.chart_view = QLabel("图表功能不可用\n请安装 PyQtChart")
            self.chart_view.setAlignment(Qt.AlignCenter)
            self.chart_view.setStyleSheet("color: gray; font-size: 14px;")
            return
        
        # 创建主图表
        self.chart = QChart()
        self.chart.setTitle("模型性能实时监控")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 创建数据系列
        self.accuracy_series = QLineSeries()
        self.accuracy_series.setName("准确率")
        self.accuracy_series.setColor(QColor(76, 175, 80))
        
        self.fps_series = QLineSeries()
        self.fps_series.setName("FPS")
        self.fps_series.setColor(QColor(33, 150, 243))
        
        self.memory_series = QLineSeries()
        self.memory_series.setName("内存使用(MB)")
        self.memory_series.setColor(QColor(255, 152, 0))
        
        # 添加系列到图表
        self.chart.addSeries(self.accuracy_series)
        self.chart.addSeries(self.fps_series)
        self.chart.addSeries(self.memory_series)
        
        # 创建坐标轴
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("时间点")
        self.x_axis.setRange(0, 100)
        
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("数值")
        self.y_axis.setRange(0, 100)
        
        self.chart.addAxis(self.x_axis, Qt.AlignBottom)
        self.chart.addAxis(self.y_axis, Qt.AlignLeft)
        
        # 附加系列到坐标轴
        self.accuracy_series.attachAxis(self.x_axis)
        self.accuracy_series.attachAxis(self.y_axis)
        self.fps_series.attachAxis(self.x_axis)
        self.fps_series.attachAxis(self.y_axis)
        self.memory_series.attachAxis(self.x_axis)
        self.memory_series.attachAxis(self.y_axis)
        
        # 设置图表视图
        self.chart_view.setChart(self.chart)
        
        # 初始化数据点计数器
        self.data_point_count = 0
    
    def connect_signals(self):
        """连接信号"""
        # 测试器信号
        self.tester.test_started.connect(self.on_test_started)
        self.tester.test_progress.connect(self.on_test_progress)
        self.tester.test_finished.connect(self.on_test_finished)
        self.tester.metrics_updated.connect(self.on_metrics_updated)
        self.tester.error_occurred.connect(self.on_error_occurred)
        
        # 按钮信号
        self.start_button.clicked.connect(self.start_test)
        self.stop_button.clicked.connect(self.stop_test)
        
        # 组合框信号
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
    
    def load_available_options(self):
        """加载可用选项"""
        # 加载模型
        models = self.tester.get_available_models()
        self.model_combo.clear()
        for name, info in models.items():
            self.model_combo.addItem(f"{name} - {info['description']}", name)
        
        # 加载数据集
        datasets = self.tester.get_available_datasets()
        self.dataset_combo.clear()
        for name, info in datasets.items():
            self.dataset_combo.addItem(f"{name} - {info['description']}", name)
        
        # 加载历史会话
        self.load_history()
    
    def load_history(self):
        """加载历史会话"""
        try:
            sessions = self.tester.get_session_history()
            self.history_table.setRowCount(len(sessions))
            
            for i, session in enumerate(sessions):
                self.history_table.setItem(i, 0, QTableWidgetItem(session.get('session_name', '')))
                self.history_table.setItem(i, 1, QTableWidgetItem(session.get('model_name', '')))
                self.history_table.setItem(i, 2, QTableWidgetItem(session.get('dataset_name', '')))
                self.history_table.setItem(i, 3, QTableWidgetItem(session.get('status', '')))
        except Exception as e:
            print(f"加载历史会话失败: {e}")
    
    def start_test(self):
        """开始测试"""
        model_name = self.model_combo.currentData()
        dataset_name = self.dataset_combo.currentData()
        session_name = self.session_edit.text().strip()
        
        if not model_name or not dataset_name:
            QMessageBox.warning(self, "警告", "请选择模型和数据集")
            return
        
        if not session_name:
            session_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 重置指标
        self.reset_metrics()
        
        # 开始测试
        self.tester.start_test(model_name, dataset_name, session_name)
    
    def stop_test(self):
        """停止测试"""
        self.tester.stop_test()
        self.on_test_finished({})
    
    def reset_metrics(self):
        """重置指标"""
        # 重置标签
        for key, label in self.metrics_labels.items():
            if key == "accuracy":
                label.setText("0.00%")
            elif key == "precision":
                label.setText("0.00%")
            elif key == "recall":
                label.setText("0.00%")
            elif key == "f1_score":
                label.setText("0.00")
            elif key == "inference_time":
                label.setText("0.00 ms")
            elif key == "fps":
                label.setText("0.00")
            elif key == "memory_usage":
                label.setText("0.00 MB")
            elif key == "detection_count":
                label.setText("0")
        
        # 重置图表
        self.accuracy_series.clear()
        self.fps_series.clear()
        self.memory_series.clear()
        self.data_point_count = 0
        
        # 重置结果文本
        self.results_text.clear()
    
    def on_test_started(self, message):
        """测试开始回调"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("测试进行中...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # 开始图表更新定时器
        self.chart_timer.start(1000)  # 每秒更新一次
    
    def on_test_progress(self, progress, message):
        """测试进度回调"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_test_finished(self, results):
        """测试完成回调"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("测试完成")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 停止图表更新定时器
        self.chart_timer.stop()
        
        # 显示结果
        if results:
            self.display_results(results)
        
        # 重新加载历史会话
        self.load_history()
    
    def on_metrics_updated(self, metrics):
        """指标更新回调"""
        # 更新指标标签
        if 'accuracy' in metrics:
            self.metrics_labels['accuracy'].setText(f"{metrics['accuracy']:.2%}")
        if 'precision' in metrics:
            self.metrics_labels['precision'].setText(f"{metrics['precision']:.2%}")
        if 'recall' in metrics:
            self.metrics_labels['recall'].setText(f"{metrics['recall']:.2%}")
        if 'f1_score' in metrics:
            self.metrics_labels['f1_score'].setText(f"{metrics['f1_score']:.2f}")
        if 'inference_time' in metrics:
            self.metrics_labels['inference_time'].setText(f"{metrics['inference_time']:.2f} ms")
        if 'memory_usage' in metrics:
            self.metrics_labels['memory_usage'].setText(f"{metrics['memory_usage']:.2f} MB")
        if 'detection_count' in metrics:
            self.metrics_labels['detection_count'].setText(str(metrics['detection_count']))
        
        # 计算FPS
        if 'inference_time' in metrics and metrics['inference_time'] > 0:
            fps = 1000 / metrics['inference_time']
            self.metrics_labels['fps'].setText(f"{fps:.2f}")
    
    def on_error_occurred(self, error_message):
        """错误发生回调"""
        QMessageBox.critical(self, "错误", error_message)
        self.on_test_finished({})
    
    def on_model_changed(self, text):
        """模型改变回调"""
        model_name = self.model_combo.currentData()
        if model_name:
            model_info = self.tester.available_models.get(model_name, {})
            info_text = f"模型: {model_name}\n"
            info_text += f"类型: {model_info.get('type', 'unknown')}\n"
            info_text += f"描述: {model_info.get('description', '无描述')}\n"
            info_text += f"路径: {model_info.get('path', '未知')}"
            self.model_info_text.setText(info_text)
    
    def on_dataset_changed(self, text):
        """数据集改变回调"""
        dataset_name = self.dataset_combo.currentData()
        if dataset_name:
            dataset_info = self.tester.available_datasets.get(dataset_name, {})
            info_text = f"数据集: {dataset_name}\n"
            info_text += f"描述: {dataset_info.get('description', '无描述')}\n"
            info_text += f"路径: {dataset_info.get('path', '未知')}"
            self.model_info_text.setText(info_text)
    
    def update_charts(self):
        """更新图表"""
        # 这里可以添加实时图表更新逻辑
        # 目前图表数据通过metrics_updated信号更新
        pass
    
    def display_results(self, results):
        """显示测试结果"""
        result_text = f"测试完成!\n\n"
        result_text += f"模型: {results.get('model_name', '未知')}\n"
        result_text += f"数据集: {results.get('dataset_name', '未知')}\n"
        result_text += f"总图像数: {results.get('total_images', 0)}\n"
        result_text += f"成功处理: {results.get('success_count', 0)}\n"
        result_text += f"失败数量: {results.get('error_count', 0)}\n"
        result_text += f"平均推理时间: {results.get('avg_inference_time', 0):.2f} ms\n"
        result_text += f"平均FPS: {results.get('fps', 0):.2f}\n"
        result_text += f"测试时长: {results.get('duration', 0):.2f} 秒\n"
        
        self.results_text.setText(result_text)


if __name__ == "__main__":
    # 测试代码
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = ModelTestUI()
    window.show()
    sys.exit(app.exec_())
