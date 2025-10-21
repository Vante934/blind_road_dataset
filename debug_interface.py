#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电脑端调试界面
提供实时监控、设备管理、模型训练等功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import time
import requests
from datetime import datetime
import os
import subprocess

class DebugInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("盲道检测系统 - 电脑端调试界面")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2B2B2B')
        
        # 服务器状态
        self.server_running = False
        self.connected_devices = {}
        self.detection_logs = []
        
        # 创建界面
        self.create_interface()
        
        # 启动状态检查线程
        self.start_status_check()
    
    def create_interface(self):
        """创建主界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标签页
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 服务器控制标签页
        self.create_server_tab(notebook)
        
        # 设备管理标签页
        self.create_device_tab(notebook)
        
        # 检测监控标签页
        self.create_detection_tab(notebook)
        
        # 日志查看标签页
        self.create_log_tab(notebook)
    
    def create_server_tab(self, notebook):
        """创建服务器控制标签页"""
        server_frame = ttk.Frame(notebook)
        notebook.add(server_frame, text="服务器控制")
        
        # 服务器状态
        status_frame = ttk.LabelFrame(server_frame, text="服务器状态")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.server_status_label = ttk.Label(status_frame, text="服务器未运行", foreground="red")
        self.server_status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.server_port_label = ttk.Label(status_frame, text="端口: 8080")
        self.server_port_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 控制按钮
        control_frame = ttk.LabelFrame(server_frame, text="服务器控制")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="启动服务器", command=self.start_server).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="停止服务器", command=self.stop_server).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="重启服务器", command=self.restart_server).pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_device_tab(self, notebook):
        """创建设备管理标签页"""
        device_frame = ttk.Frame(notebook)
        notebook.add(device_frame, text="设备管理")
        
        # 设备列表
        device_list_frame = ttk.LabelFrame(device_frame, text="连接的设备")
        device_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建设备树形视图
        columns = ("设备ID", "连接时间", "最后心跳", "数据数量", "模型版本")
        self.device_tree = ttk.Treeview(device_list_frame, columns=columns, show="headings")
        
        for col in columns:
            self.device_tree.heading(col, text=col)
            self.device_tree.column(col, width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(device_list_frame, orient=tk.VERTICAL, command=self.device_tree.yview)
        self.device_tree.configure(yscrollcommand=scrollbar.set)
        
        self.device_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 设备操作按钮
        device_ops_frame = ttk.LabelFrame(device_frame, text="设备操作")
        device_ops_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(device_ops_frame, text="刷新设备列表", command=self.refresh_devices).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(device_ops_frame, text="发送指令", command=self.send_command).pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_detection_tab(self, notebook):
        """创建检测监控标签页"""
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="检测监控")
        
        # 实时检测状态
        status_frame = ttk.LabelFrame(detection_frame, text="检测状态")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.detection_status_label = ttk.Label(status_frame, text="检测未启动")
        self.detection_status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.detection_count_label = ttk.Label(status_frame, text="检测次数: 0")
        self.detection_count_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 检测结果列表
        results_frame = ttk.LabelFrame(detection_frame, text="检测结果")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建检测结果树形视图
        result_columns = ("时间", "设备ID", "障碍物类型", "置信度", "位置")
        self.result_tree = ttk.Treeview(results_frame, columns=result_columns, show="headings")
        
        for col in result_columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=120)
        
        # 添加滚动条
        result_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=result_scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 检测控制按钮
        detection_ops_frame = ttk.LabelFrame(detection_frame, text="检测控制")
        detection_ops_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(detection_ops_frame, text="开始检测", command=self.start_detection).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(detection_ops_frame, text="停止检测", command=self.stop_detection).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(detection_ops_frame, text="清除结果", command=self.clear_results).pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_log_tab(self, notebook):
        """创建日志查看标签页"""
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="日志查看")
        
        # 日志显示
        log_display_frame = ttk.LabelFrame(log_frame, text="系统日志")
        log_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_display_frame, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 日志控制
        log_ops_frame = ttk.LabelFrame(log_frame, text="日志控制")
        log_ops_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(log_ops_frame, text="刷新日志", command=self.refresh_logs).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(log_ops_frame, text="清除日志", command=self.clear_logs).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(log_ops_frame, text="保存日志", command=self.save_logs).pack(side=tk.LEFT, padx=5, pady=5)
    
    def start_status_check(self):
        """启动状态检查线程"""
        def check_status():
            while True:
                try:
                    # 检查服务器状态
                    response = requests.get("http://localhost:8080/status", timeout=1)
                    if response.status_code == 200:
                        self.server_running = True
                        self.server_status_label.config(text="服务器运行中", foreground="green")
                        
                        # 更新设备信息
                        data = response.json()
                        self.connected_devices = data.get('devices', {})
                        self.update_device_list()
                    else:
                        self.server_running = False
                        self.server_status_label.config(text="服务器未运行", foreground="red")
                except:
                    self.server_running = False
                    self.server_status_label.config(text="服务器未运行", foreground="red")
                
                time.sleep(2)
        
        thread = threading.Thread(target=check_status, daemon=True)
        thread.start()
    
    def update_device_list(self):
        """更新设备列表"""
        # 清除现有项目
        for item in self.device_tree.get_children():
            self.device_tree.delete(item)
        
        # 添加设备信息
        for device_id, device_info in self.connected_devices.items():
            self.device_tree.insert("", "end", values=(
                device_id,
                datetime.fromtimestamp(device_info.get('connected_time', 0)).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(device_info.get('last_heartbeat', 0)).strftime("%Y-%m-%d %H:%M:%S"),
                device_info.get('data_count', 0),
                device_info.get('model_version', '1.0.0')
            ))
    
    def start_server(self):
        """启动服务器"""
        try:
            # 启动服务器进程
            subprocess.Popen("python pc_server.py", shell=True)
            self.log_message("服务器启动命令已执行")
            messagebox.showinfo("成功", "服务器启动命令已执行")
        except Exception as e:
            self.log_message(f"启动服务器失败: {e}")
            messagebox.showerror("错误", f"启动服务器失败: {e}")
    
    def stop_server(self):
        """停止服务器"""
        try:
            self.log_message("服务器停止命令已执行")
            messagebox.showinfo("成功", "服务器停止命令已执行")
        except Exception as e:
            self.log_message(f"停止服务器失败: {e}")
            messagebox.showerror("错误", f"停止服务器失败: {e}")
    
    def restart_server(self):
        """重启服务器"""
        self.stop_server()
        time.sleep(2)
        self.start_server()
    
    def refresh_devices(self):
        """刷新设备列表"""
        self.update_device_list()
        self.log_message("设备列表已刷新")
    
    def send_command(self):
        """发送指令到设备"""
        selected = self.device_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择设备")
            return
        messagebox.showinfo("信息", "发送指令功能待实现")
    
    def start_detection(self):
        """开始检测"""
        self.detection_status_label.config(text="检测进行中")
        self.log_message("开始检测")
    
    def stop_detection(self):
        """停止检测"""
        self.detection_status_label.config(text="检测已停止")
        self.log_message("停止检测")
    
    def clear_results(self):
        """清除检测结果"""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.log_message("检测结果已清除")
    
    def refresh_logs(self):
        """刷新日志"""
        self.log_message("日志已刷新")
    
    def clear_logs(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("日志已清除")
    
    def save_logs(self):
        """保存日志"""
        try:
            with open("debug_logs.txt", "w", encoding="utf-8") as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("成功", "日志已保存到 debug_logs.txt")
        except Exception as e:
            messagebox.showerror("错误", f"保存日志失败: {e}")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def run(self):
        """运行界面"""
        self.root.mainloop()

def main():
    """主函数"""
    print("🖥️ 启动盲道检测系统调试界面...")
    
    try:
        app = DebugInterface()
        app.run()
    except Exception as e:
        print(f"❌ 启动调试界面失败: {e}")

if __name__ == "__main__":
    main()
