#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统状态检查脚本
检查盲道障碍检测系统的各个组件状态
"""

import os
import sys
import json
import importlib
from pathlib import Path

def check_python_environment():
    """检查Python环境"""
    print("🐍 检查Python环境...")
    
    try:
        import sys
        python_version = sys.version_info
        print(f"  ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("  ⚠️ 建议使用Python 3.8或更高版本")
            return False
        return True
    except Exception as e:
        print(f"  ❌ Python环境检查失败: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        'cv2', 'numpy', 'PyQt5', 'ultralytics', 'requests', 'json'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"  ✅ OpenCV: {cv2.__version__}")
            elif package == 'numpy':
                import numpy
                print(f"  ✅ NumPy: {numpy.__version__}")
            elif package == 'PyQt5':
                import PyQt5
                print(f"  ✅ PyQt5: 已安装")
            elif package == 'ultralytics':
                import ultralytics
                print(f"  ✅ Ultralytics: {ultralytics.__version__}")
            elif package == 'requests':
                import requests
                print(f"  ✅ Requests: {requests.__version__}")
            elif package == 'json':
                import json
                print(f"  ✅ JSON: 内置模块")
        except ImportError:
            print(f"  ❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  ⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("  请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_file_structure():
    """检查文件结构"""
    print("\n📁 检查文件结构...")
    
    required_files = [
        'core/enhanced_mobile_app.py',
        'modules/environment_detector.py',
        'modules/trajectory_predictor.py',
        'enhanced_annotation_tool.py',
        'environment_annotation_classes.json',
        'environment_detection_config.json',
        'test_environment_detection.py',
        'prepare_environment_training_data.py',
        'enhanced_environment_detector.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n  ⚠️ 缺少文件: {len(missing_files)} 个")
        return False
    
    return True

def check_models():
    """检查模型文件"""
    print("\n🤖 检查模型文件...")
    
    model_files = [
        'models/yolov8n.pt',
        'models/yolo11n.pt',
        'yolov8n.pt'
    ]
    
    found_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✅ {model_file} ({size:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"  ❌ {model_file}")
    
    if not found_models:
        print("  ⚠️ 没有找到YOLO模型文件")
        print("  请下载YOLO模型文件到models目录")
        return False
    
    return True

def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 检查配置文件...")
    
    config_files = [
        'environment_annotation_classes.json',
        'environment_detection_config.json',
        'voice_config.json',
        'sdk_config.json'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"  ✅ {config_file} (格式正确)")
            except json.JSONDecodeError:
                print(f"  ⚠️ {config_file} (JSON格式错误)")
            except Exception as e:
                print(f"  ❌ {config_file} (读取失败: {e})")
        else:
            print(f"  ❌ {config_file}")

def check_environment_detection():
    """检查环境检测模块"""
    print("\n🌍 检查环境检测模块...")
    
    try:
        from modules.environment_detector import EnvironmentDetector
        detector = EnvironmentDetector()
        print("  ✅ 环境检测模块加载成功")
        
        # 测试检测功能
        import numpy as np
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = detector.detect_environment(test_frame, [])
        
        if 'overall_safety_level' in result:
            print("  ✅ 环境检测功能正常")
            return True
        else:
            print("  ⚠️ 环境检测功能异常")
            return False
            
    except ImportError as e:
        print(f"  ❌ 环境检测模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 环境检测模块测试失败: {e}")
        return False

def check_annotation_tool():
    """检查标注工具"""
    print("\n🏷️ 检查标注工具...")
    
    try:
        # 检查标注类别文件
        if os.path.exists('environment_annotation_classes.json'):
            with open('environment_annotation_classes.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                classes = data.get('environment_annotation_classes', {})
                total_classes = sum(len(category) for category in classes.values())
                print(f"  ✅ 标注类别文件正常 ({total_classes} 个类别)")
        else:
            print("  ❌ 标注类别文件不存在")
            return False
        
        # 检查标注工具文件
        if os.path.exists('enhanced_annotation_tool.py'):
            print("  ✅ 标注工具文件存在")
        else:
            print("  ❌ 标注工具文件不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 标注工具检查失败: {e}")
        return False

def generate_system_report():
    """生成系统报告"""
    print("\n📊 生成系统状态报告...")
    
    report = {
        'timestamp': __import__('time').time(),
        'python_version': sys.version,
        'system_status': {
            'python_environment': check_python_environment(),
            'dependencies': check_dependencies(),
            'file_structure': check_file_structure(),
            'models': check_models(),
            'environment_detection': check_environment_detection(),
            'annotation_tool': check_annotation_tool()
        }
    }
    
    # 计算总体状态
    status_values = list(report['system_status'].values())
    overall_status = all(status_values)
    report['overall_status'] = overall_status
    
    # 保存报告
    with open('system_status_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 系统状态报告已保存: system_status_report.json")
    
    return report

def main():
    """主函数"""
    print("🔍 盲道障碍检测系统状态检查")
    print("=" * 50)
    
    # 检查各个组件
    python_ok = check_python_environment()
    deps_ok = check_dependencies()
    files_ok = check_file_structure()
    models_ok = check_models()
    env_ok = check_environment_detection()
    annotation_ok = check_annotation_tool()
    
    # 检查配置文件
    check_config_files()
    
    # 生成报告
    report = generate_system_report()
    
    # 显示总结
    print("\n" + "=" * 50)
    print("📋 系统状态总结")
    print("=" * 50)
    
    components = [
        ("Python环境", python_ok),
        ("依赖包", deps_ok),
        ("文件结构", files_ok),
        ("模型文件", models_ok),
        ("环境检测", env_ok),
        ("标注工具", annotation_ok)
    ]
    
    for name, status in components:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    overall_status = all([python_ok, deps_ok, files_ok, models_ok, env_ok, annotation_ok])
    
    print("\n" + "=" * 50)
    if overall_status:
        print("🎉 系统状态良好，所有组件正常！")
        print("💡 您可以正常使用所有功能")
    else:
        print("⚠️ 系统存在问题，请检查上述错误")
        print("💡 建议先解决标记为❌的问题")
    
    print("=" * 50)
    
    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
