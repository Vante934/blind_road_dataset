#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时图表模块
提供模型测试的实时图表显示和更新功能
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
try:
    from PyQt5.QtChart import (QChart, QChartView, QLineSeries, QScatterSeries, 
                               QValueAxis, QDateTimeAxis, QLegend, QAbstractSeries)
    CHART_AVAILABLE = True
except ImportError:
    try:
        from PyQt5.QtCharts import (QChart, QChartView, QLineSeries, QScatterSeries, 
                                   QValueAxis, QDateTimeAxis, QLegend, QAbstractSeries)
        CHART_AVAILABLE = True
    except ImportError:
        CHART_AVAILABLE = False
        print("⚠️ PyQt5.QtChart 不可用，图表功能将被禁用")

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class RealTimeChart(QWidget):
    """实时图表组件"""
    
    data_updated = pyqtSignal(dict)
    
    def __init__(self, title: str = "实时图表", max_points: int = 100):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.data_points = deque(maxlen=max_points)
        self.time_points = deque(maxlen=max_points)
        
        self.init_ui()
        self.setup_chart()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 图表视图
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 清除按钮
        self.clear_button = QPushButton("清除数据")
        self.clear_button.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_button)
        
        # 暂停/继续按钮
        self.pause_button = QPushButton("暂停")
        self.pause_button.setCheckable(True)
        self.pause_button.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_button)
        
        # 状态标签
        self.status_label = QLabel("运行中")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
    def setup_chart(self):
        """设置图表"""
        self.chart = QChart()
        self.chart.setTitle(self.title)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # 创建坐标轴
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("时间点")
        self.x_axis.setRange(0, self.max_points)
        
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("数值")
        self.y_axis.setRange(0, 100)
        
        self.chart.addAxis(self.x_axis, Qt.AlignBottom)
        self.chart.addAxis(self.y_axis, Qt.AlignLeft)
        
        self.chart_view.setChart(self.chart)
        
        # 初始化数据点计数器
        self.point_count = 0
        self.is_paused = False
        
    def add_data_point(self, value: float, timestamp: datetime = None):
        """添加数据点"""
        if self.is_paused:
            return
            
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data_points.append(value)
        self.time_points.append(timestamp)
        self.point_count += 1
        
        # 更新图表
        self.update_chart()
        
        # 发送数据更新信号
        self.data_updated.emit({
            'value': value,
            'timestamp': timestamp,
            'point_count': self.point_count
        })
    
    def update_chart(self):
        """更新图表"""
        if not self.data_points:
            return
        
        # 清除现有系列
        self.chart.removeAllSeries()
        
        # 创建新的数据系列
        series = QLineSeries()
        series.setName("实时数据")
        series.setColor(QColor(33, 150, 243))
        series.setPenWidth(2)
        
        # 添加数据点
        for i, (value, timestamp) in enumerate(zip(self.data_points, self.time_points)):
            series.append(i, value)
        
        # 添加到图表
        self.chart.addSeries(series)
        series.attachAxis(self.x_axis)
        series.attachAxis(self.y_axis)
        
        # 更新坐标轴范围
        if len(self.data_points) > 0:
            min_val = min(self.data_points)
            max_val = max(self.data_points)
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 1
            
            self.y_axis.setRange(min_val - margin, max_val + margin)
            self.x_axis.setRange(max(0, len(self.data_points) - self.max_points), len(self.data_points))
    
    def clear_data(self):
        """清除数据"""
        self.data_points.clear()
        self.time_points.clear()
        self.point_count = 0
        self.update_chart()
        self.status_label.setText("数据已清除")
    
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText("继续")
            self.status_label.setText("已暂停")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.pause_button.setText("暂停")
            self.status_label.setText("运行中")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.data_points:
            return {}
        
        data_array = np.array(self.data_points)
        return {
            'count': len(self.data_points),
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array)
        }


class MultiMetricChart(QWidget):
    """多指标图表组件"""
    
    def __init__(self, title: str = "多指标监控", max_points: int = 100):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.metrics_data = {}
        self.time_points = deque(maxlen=max_points)
        
        self.init_ui()
        self.setup_chart()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 图表视图
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 指标选择
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["准确率", "FPS", "内存使用", "推理时间"])
        control_layout.addWidget(QLabel("显示指标:"))
        control_layout.addWidget(self.metric_combo)
        
        # 清除按钮
        self.clear_button = QPushButton("清除数据")
        self.clear_button.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
    def setup_chart(self):
        """设置图表"""
        self.chart = QChart()
        self.chart.setTitle(self.title)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # 创建坐标轴
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("时间点")
        self.x_axis.setRange(0, self.max_points)
        
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("数值")
        self.y_axis.setRange(0, 100)
        
        self.chart.addAxis(self.x_axis, Qt.AlignBottom)
        self.chart.addAxis(self.y_axis, Qt.AlignLeft)
        
        self.chart_view.setChart(self.chart)
        
        # 初始化数据点计数器
        self.point_count = 0
        
        # 定义指标配置
        self.metric_configs = {
            "准确率": {"color": QColor(76, 175, 80), "unit": "%"},
            "FPS": {"color": QColor(33, 150, 243), "unit": ""},
            "内存使用": {"color": QColor(255, 152, 0), "unit": "MB"},
            "推理时间": {"color": QColor(156, 39, 176), "unit": "ms"}
        }
        
    def add_metrics(self, metrics: Dict, timestamp: datetime = None):
        """添加指标数据"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # 存储时间点
        self.time_points.append(timestamp)
        
        # 存储指标数据
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_data:
                self.metrics_data[metric_name] = deque(maxlen=self.max_points)
            self.metrics_data[metric_name].append(value)
        
        self.point_count += 1
        
        # 更新图表
        self.update_chart()
    
    def update_chart(self):
        """更新图表"""
        # 清除现有系列
        self.chart.removeAllSeries()
        
        # 为每个指标创建系列
        for metric_name, data in self.metrics_data.items():
            if not data:
                continue
                
            series = QLineSeries()
            series.setName(metric_name)
            
            # 设置颜色
            if metric_name in self.metric_configs:
                series.setColor(self.metric_configs[metric_name]["color"])
            
            # 添加数据点
            for i, value in enumerate(data):
                series.append(i, value)
            
            # 添加到图表
            self.chart.addSeries(series)
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)
        
        # 更新坐标轴范围
        if self.time_points:
            self.x_axis.setRange(max(0, len(self.time_points) - self.max_points), len(self.time_points))
            
            # 计算Y轴范围
            all_values = []
            for data in self.metrics_data.values():
                all_values.extend(data)
            
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.1 if max_val != min_val else 1
                self.y_axis.setRange(min_val - margin, max_val + margin)
    
    def clear_data(self):
        """清除数据"""
        self.metrics_data.clear()
        self.time_points.clear()
        self.point_count = 0
        self.update_chart()
    
    def get_metric_statistics(self, metric_name: str) -> Dict:
        """获取指定指标的统计信息"""
        if metric_name not in self.metrics_data or not self.metrics_data[metric_name]:
            return {}
        
        data_array = np.array(self.metrics_data[metric_name])
        return {
            'metric_name': metric_name,
            'count': len(data_array),
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array)
        }


class PerformanceDashboard(QWidget):
    """性能仪表板"""
    
    def __init__(self):
        super().__init__()
        self.charts = {}
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建多个图表
        self.accuracy_chart = RealTimeChart("准确率监控", max_points=50)
        self.fps_chart = RealTimeChart("FPS监控", max_points=50)
        self.memory_chart = RealTimeChart("内存使用监控", max_points=50)
        self.multi_chart = MultiMetricChart("综合指标监控", max_points=50)
        
        # 添加到布局
        layout.addWidget(self.accuracy_chart)
        layout.addWidget(self.fps_chart)
        layout.addWidget(self.memory_chart)
        layout.addWidget(self.multi_chart)
        
        # 存储图表引用
        self.charts = {
            'accuracy': self.accuracy_chart,
            'fps': self.fps_chart,
            'memory': self.memory_chart,
            'multi': self.multi_chart
        }
    
    def update_metrics(self, metrics: Dict):
        """更新指标"""
        # 更新单个指标图表
        if 'accuracy' in metrics:
            self.accuracy_chart.add_data_point(metrics['accuracy'] * 100)  # 转换为百分比
        
        if 'fps' in metrics:
            self.fps_chart.add_data_point(metrics['fps'])
        
        if 'memory_usage' in metrics:
            self.memory_chart.add_data_point(metrics['memory_usage'])
        
        # 更新多指标图表
        self.multi_chart.add_metrics(metrics)
    
    def clear_all_data(self):
        """清除所有数据"""
        for chart in self.charts.values():
            if hasattr(chart, 'clear_data'):
                chart.clear_data()
    
    def get_all_statistics(self) -> Dict:
        """获取所有统计信息"""
        stats = {}
        for name, chart in self.charts.items():
            if hasattr(chart, 'get_statistics'):
                stats[name] = chart.get_statistics()
        return stats


if __name__ == "__main__":
    # 测试代码
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # 测试实时图表
    chart = RealTimeChart("测试图表")
    chart.show()
    
    # 模拟数据更新
    timer = QTimer()
    import random
    counter = 0
    
    def update_data():
        global counter
        value = 50 + 30 * np.sin(counter * 0.1) + random.uniform(-5, 5)
        chart.add_data_point(value)
        counter += 1
    
    timer.timeout.connect(update_data)
    timer.start(100)  # 每100ms更新一次
    
    sys.exit(app.exec_())
