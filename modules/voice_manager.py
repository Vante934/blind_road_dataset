# -*- coding: utf-8 -*-
"""
语音库管理工具
提供GUI界面来编辑和管理语音提示文本
"""

import sys
import json
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from modules.voice_library import VoiceLibrary

class VoiceManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.voice_library = VoiceLibrary()
        self.initUI()
        self.load_data()
    
    def initUI(self):
        self.setWindowTitle("语音库管理工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        layout = QHBoxLayout(central_widget)
        
        # 左侧面板 - 障碍物类型列表
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 1)
        
        # 右侧面板 - 编辑区域
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 2)
        
        # 创建菜单栏
        self.create_menu_bar()
    
    def create_left_panel(self):
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("障碍物类型")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # 类别选择
        self.category_combo = QComboBox()
        # 从配置文件中动态加载类别
        self.load_categories()
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        layout.addWidget(QLabel("类别:"))
        layout.addWidget(self.category_combo)
        
        # 障碍物列表
        self.obstacle_list = QListWidget()
        self.obstacle_list.itemClicked.connect(self.on_obstacle_selected)
        layout.addWidget(QLabel("障碍物类型:"))
        layout.addWidget(self.obstacle_list)
        
        # 添加新障碍物按钮
        add_btn = QPushButton("添加新障碍物")
        add_btn.clicked.connect(self.add_new_obstacle)
        layout.addWidget(add_btn)
        
        # 类别映射
        layout.addWidget(QLabel("类别映射:"))
        self.mapping_list = QListWidget()
        layout.addWidget(self.mapping_list)
        
        # 添加映射按钮
        add_mapping_btn = QPushButton("添加类别映射")
        add_mapping_btn.clicked.connect(self.add_class_mapping)
        layout.addWidget(add_mapping_btn)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("语音模板编辑")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # 文本粘贴/智能识别区域
        paste_group = QGroupBox("文本粘贴/智能识别")
        paste_layout = QVBoxLayout(paste_group)
        self.paste_text = QTextEdit()
        self.paste_text.setPlaceholderText("请粘贴或输入描述性文本，如‘前方有汽车鸣笛，请注意安全’……")
        paste_layout.addWidget(self.paste_text)
        smart_btn = QPushButton("智能识别并填充模板")
        smart_btn.clicked.connect(self.smart_recognize_text)
        paste_layout.addWidget(smart_btn)
        layout.addWidget(paste_group)
        
        # 当前编辑的障碍物信息
        self.current_obstacle_label = QLabel("请选择障碍物类型")
        layout.addWidget(self.current_obstacle_label)
        
        # 模板编辑区域
        template_group = QGroupBox("语音模板")
        template_layout = QVBoxLayout(template_group)
        
        # 远距离模板
        far_layout = QHBoxLayout()
        far_layout.addWidget(QLabel("远距离(3-5米):"))
        self.far_template = QLineEdit()
        far_layout.addWidget(self.far_template)
        template_layout.addLayout(far_layout)
        
        # 近距离模板
        near_layout = QHBoxLayout()
        near_layout.addWidget(QLabel("近距离(1-2米):"))
        self.near_template = QLineEdit()
        near_layout.addWidget(self.near_template)
        template_layout.addLayout(near_layout)
        
        # 危险距离模板
        danger_layout = QHBoxLayout()
        danger_layout.addWidget(QLabel("危险距离(0.5-1米):"))
        self.danger_template = QLineEdit()
        danger_layout.addWidget(self.danger_template)
        template_layout.addLayout(danger_layout)
        
        # 紧急距离模板
        emergency_layout = QHBoxLayout()
        emergency_layout.addWidget(QLabel("紧急距离(<0.5米):"))
        self.emergency_template = QLineEdit()
        emergency_layout.addWidget(self.emergency_template)
        template_layout.addLayout(emergency_layout)
        
        layout.addWidget(template_group)
        
        # 保存按钮
        save_btn = QPushButton("保存更改")
        save_btn.clicked.connect(self.save_changes)
        layout.addWidget(save_btn)
        
        # 测试按钮
        test_btn = QPushButton("测试语音")
        test_btn.clicked.connect(self.test_voice)
        layout.addWidget(test_btn)
        
        # 特殊场景编辑
        scenario_group = QGroupBox("特殊场景")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_list = QListWidget()
        scenario_layout.addWidget(self.scenario_list)
        
        add_scenario_btn = QPushButton("添加特殊场景")
        add_scenario_btn.clicked.connect(self.add_special_scenario)
        scenario_layout.addWidget(add_scenario_btn)
        
        layout.addWidget(scenario_group)
        
        layout.addStretch()
        return panel
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        save_action = QAction('保存配置', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        reload_action = QAction('重新加载', self)
        reload_action.setShortcut('Ctrl+R')
        reload_action.triggered.connect(self.reload_config)
        file_menu.addAction(reload_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        export_action = QAction('导出配置', self)
        export_action.triggered.connect(self.export_config)
        tools_menu.addAction(export_action)
        
        import_action = QAction('导入配置', self)
        import_action.triggered.connect(self.import_config)
        tools_menu.addAction(import_action)
    
    def load_categories(self):
        """从配置文件加载类别"""
        categories = []
        for main_category in self.voice_library.obstacle_types.keys():
            categories.append(main_category)
        self.category_combo.addItems(categories)
    
    def load_data(self):
        """加载数据"""
        self.on_category_changed(self.category_combo.currentText())
        self.load_mappings()
        self.load_scenarios()
    
    def on_category_changed(self, category):
        """类别改变时更新障碍物列表"""
        self.obstacle_list.clear()
        
        # 处理新的嵌套结构
        def add_obstacles_recursive(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'name' in value and 'templates' in value:
                        # 这是一个障碍物定义
                        display_name = f"{prefix}{key} ({value['name']})" if prefix else f"{key} ({value['name']})"
                        item = QListWidgetItem(display_name)
                        item.setData(Qt.UserRole, f"{prefix}{key}" if prefix else key)
                        self.obstacle_list.addItem(item)
                    else:
                        # 这是一个子类别，递归处理
                        new_prefix = f"{prefix}{key}." if prefix else f"{key}."
                        add_obstacles_recursive(value, new_prefix)
        
        # 查找匹配的类别
        if category in self.voice_library.obstacle_types:
            add_obstacles_recursive(self.voice_library.obstacle_types[category])
        else:
            # 尝试在新结构中查找
            for main_category, subcategories in self.voice_library.obstacle_types.items():
                if category in subcategories:
                    add_obstacles_recursive(subcategories[category])
                    break
    
    def on_obstacle_selected(self, item):
        """选择障碍物时加载模板"""
        category = self.category_combo.currentText()
        obstacle_type = item.data(Qt.UserRole)
        
        # 递归查找障碍物信息
        def find_obstacle_info(data, target_type):
            for key, value in data.items():
                if key == target_type and isinstance(value, dict) and 'name' in value and 'templates' in value:
                    return value
                elif isinstance(value, dict) and 'name' not in value:
                    # 这是一个子类别，递归查找
                    result = find_obstacle_info(value, target_type)
                    if result:
                        return result
            return None
        
        # 查找障碍物信息
        obstacle_info = None
        if category in self.voice_library.obstacle_types:
            obstacle_info = find_obstacle_info(self.voice_library.obstacle_types[category], obstacle_type)
        
        if obstacle_info:
            templates = obstacle_info['templates']
            
            self.current_obstacle_label.setText(f"当前编辑: {category}.{obstacle_type} ({obstacle_info['name']})")
            
            self.far_template.setText(templates.get('far', ''))
            self.near_template.setText(templates.get('near', ''))
            self.danger_template.setText(templates.get('danger', ''))
            self.emergency_template.setText(templates.get('emergency', ''))
    
    def add_new_obstacle(self):
        """添加新障碍物"""
        category = self.category_combo.currentText()
        
        name, ok = QInputDialog.getText(self, "添加障碍物", "障碍物名称:")
        if ok and name:
            obstacle_type, ok = QInputDialog.getText(self, "添加障碍物", "障碍物类型标识:")
            if ok and obstacle_type:
                # 创建默认模板
                templates = {
                    'far': f"前方{{distance}}米{{direction}}有{name}",
                    'near': f"{{direction}}{{distance}}米有{name}，请减速",
                    'danger': f"危险！{{direction}}{{distance}}米有{name}，立即减速！",
                    'emergency': f"危险！立即停止！{{direction}}{{distance}}米有{name}"
                }
                
                self.voice_library.add_custom_obstacle(category, obstacle_type, name, templates)
                self.on_category_changed(category)
    
    def load_mappings(self):
        """加载类别映射"""
        self.mapping_list.clear()
        for class_id, mapping in self.voice_library.class_mapping.items():
            # 处理新的映射格式（4层结构）和旧格式（2层结构）
            if isinstance(mapping, list) and len(mapping) >= 2:
                if len(mapping) == 2:
                    # 旧格式: [category, obstacle_type]
                    category, obstacle_type = mapping
                    display_text = f"class{class_id} -> {category}.{obstacle_type}"
                else:
                    # 新格式: [category1, category2, category3, obstacle_type]
                    category = mapping[0]
                    obstacle_type = mapping[-1]  # 最后一个元素是障碍物类型
                    display_text = f"class{class_id} -> {'.'.join(mapping)}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, class_id)
                self.mapping_list.addItem(item)
    
    def add_class_mapping(self):
        """添加类别映射"""
        class_id, ok = QInputDialog.getText(self, "添加映射", "类别ID:")
        if ok and class_id:
            category, ok = QInputDialog.getItem(self, "添加映射", "选择类别:", 
                                              ["static", "dynamic", "ground"])
            if ok and category:
                obstacle_type, ok = QInputDialog.getText(self, "添加映射", "障碍物类型:")
                if ok and obstacle_type:
                    self.voice_library.add_class_mapping(class_id, category, obstacle_type)
                    self.load_mappings()
    
    def load_scenarios(self):
        """加载特殊场景"""
        self.scenario_list.clear()
        for scenario_name, message in self.voice_library.special_scenarios.items():
            item = QListWidgetItem(f"{scenario_name}: {message}")
            item.setData(Qt.UserRole, scenario_name)
            self.scenario_list.addItem(item)
    
    def add_special_scenario(self):
        """添加特殊场景"""
        scenario_name, ok = QInputDialog.getText(self, "添加特殊场景", "场景名称:")
        if ok and scenario_name:
            message, ok = QInputDialog.getText(self, "添加特殊场景", "语音消息:")
            if ok and message:
                self.voice_library.add_special_scenario(scenario_name, message)
                self.load_scenarios()
    
    def save_changes(self):
        """保存更改"""
        category = self.category_combo.currentText()
        current_item = self.obstacle_list.currentItem()
        
        if current_item:
            obstacle_type = current_item.data(Qt.UserRole)
            
            templates = {
                'far': self.far_template.text(),
                'near': self.near_template.text(),
                'danger': self.danger_template.text(),
                'emergency': self.emergency_template.text()
            }
            
            for level, template in templates.items():
                self.voice_library.update_obstacle_template(category, obstacle_type, level, template)
            
            QMessageBox.information(self, "保存成功", "语音模板已保存")
    
    def save_config(self):
        """保存配置到文件"""
        if self.voice_library.save_config():
            QMessageBox.information(self, "保存成功", "配置已保存到文件")
        else:
            QMessageBox.warning(self, "保存失败", "配置保存失败")
    
    def reload_config(self):
        """重新加载配置"""
        self.voice_library.reload_config()
        self.load_data()
        QMessageBox.information(self, "重新加载", "配置已重新加载")
    
    def test_voice(self):
        """测试语音"""
        # 这里可以添加语音测试功能
        QMessageBox.information(self, "测试", "语音测试功能待实现")
    
    def export_config(self):
        """导出配置"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出配置", "voice_config_export.json", "JSON Files (*.json)")
        if file_path:
            try:
                config = {
                    'obstacle_types': self.voice_library.obstacle_types,
                    'class_mapping': self.voice_library.class_mapping,
                    'generic_templates': self.voice_library.generic_templates,
                    'special_scenarios': self.voice_library.special_scenarios
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "导出成功", f"配置已导出到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"导出失败: {e}")
    
    def import_config(self):
        """导入配置"""
        file_path, _ = QFileDialog.getOpenFileName(self, "导入配置", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.voice_library.obstacle_types = config.get('obstacle_types', {})
                self.voice_library.class_mapping = config.get('class_mapping', {})
                self.voice_library.generic_templates = config.get('generic_templates', {})
                self.voice_library.special_scenarios = config.get('special_scenarios', {})
                
                self.load_data()
                QMessageBox.information(self, "导入成功", "配置已导入")
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"导入失败: {e}")

    def smart_recognize_text(self):
        """智能识别文本内容并自动分类、填充模板"""
        text = self.paste_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先粘贴或输入文本！")
            return
        # 简单关键词分类（可扩展为NLP）
        mapping = {
            "汽车鸣笛": ("env_sound", "car_horn"),
            "施工噪音": ("env_sound", "construction_noise"),
            "警报": ("env_sound", "alarm"),
            "行人": ("dynamic", "person"),
            "车辆": ("static", "vehicle"),
            "坑洼": ("ground", "pothole"),
            "台阶": ("ground", "step"),
            "斜坡": ("ground", "slope"),
            "积水": ("ground", "water"),
            "摊位": ("static", "stall"),
            "垃圾桶": ("static", "trash_bin"),
            "宠物": ("dynamic", "pet"),
            "家具": ("static", "furniture"),
            "移动车辆": ("dynamic", "moving_vehicle")
        }
        found = False
        for k, (cat, typ) in mapping.items():
            if k in text:
                self.category_combo.setCurrentText(cat)
                self.on_category_changed(cat)
                # 自动选中障碍物类型
                for i in range(self.obstacle_list.count()):
                    item = self.obstacle_list.item(i)
                    if typ in item.text():
                        self.obstacle_list.setCurrentItem(item)
                        self.on_obstacle_selected(item)
                        break
                # 自动填充模板
                self.far_template.setText(text)
                self.near_template.setText(text)
                self.danger_template.setText(text)
                self.emergency_template.setText(text)
                found = True
                break
        if not found:
            QMessageBox.information(self, "未识别", "未能识别文本内容，请手动选择类别和类型。")

def main():
    app = QApplication(sys.argv)
    window = VoiceManagerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 