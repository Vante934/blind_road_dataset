#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理机制 - 统一的错误处理和恢复策略
提供完善的错误分类、处理、记录和恢复机制
"""

import logging
import traceback
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from functools import wraps
import json

# ==================== 错误类型定义 ====================

class ErrorType(str, Enum):
    """错误类型枚举"""
    VALIDATION_ERROR = "ValidationError"
    AUTHENTICATION_ERROR = "AuthenticationError"
    AUTHORIZATION_ERROR = "AuthorizationError"
    BUSINESS_ERROR = "BusinessError"
    SYSTEM_ERROR = "SystemError"
    NETWORK_ERROR = "NetworkError"
    TIMEOUT_ERROR = "TimeoutError"
    RESOURCE_ERROR = "ResourceError"
    CONFIGURATION_ERROR = "ConfigurationError"
    DEPENDENCY_ERROR = "DependencyError"

class ErrorSeverity(str, Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """错误分类枚举"""
    CLIENT_ERROR = "client_error"      # 客户端错误
    SERVER_ERROR = "server_error"     # 服务器错误
    NETWORK_ERROR = "network_error"   # 网络错误
    DATA_ERROR = "data_error"         # 数据错误
    SYSTEM_ERROR = "system_error"     # 系统错误

# ==================== 错误数据类 ====================

@dataclass
class ErrorContext:
    """错误上下文信息"""
    error_id: str
    timestamp: str
    module: str
    function: str
    line_number: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: ErrorType
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: str
    context: ErrorContext
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_after: Optional[int] = None

@dataclass
class RecoveryAction:
    """恢复动作"""
    action_type: str
    parameters: Dict[str, Any]
    success: bool = False
    error_message: Optional[str] = None

# ==================== 错误处理器 ====================

class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorType, Callable] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # 注册默认恢复策略
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self):
        """注册默认恢复策略"""
        self.recovery_strategies = {
            ErrorType.NETWORK_ERROR: self._handle_network_error,
            ErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            ErrorType.RESOURCE_ERROR: self._handle_resource_error,
            ErrorType.DEPENDENCY_ERROR: self._handle_dependency_error,
            ErrorType.SYSTEM_ERROR: self._handle_system_error
        }
    
    def handle_error(self, 
                    error: Exception, 
                    error_type: ErrorType,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    module: str = "unknown",
                    additional_data: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """处理错误"""
        
        # 创建错误上下文
        context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            module=module,
            function=error.__class__.__name__,
            line_number=getattr(error, '__traceback__', None) and error.__traceback__.tb_lineno or 0,
            additional_data=additional_data
        )
        
        # 确定错误分类
        category = self._categorize_error(error_type)
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type=error_type,
            severity=severity,
            category=category,
            message=str(error),
            details=self._get_error_details(error),
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        # 记录错误
        self._log_error(error_info)
        
        # 添加到历史记录
        self.error_history.append(error_info)
        
        # 尝试恢复
        recovery_action = self._attempt_recovery(error_info)
        
        return error_info
    
    def _categorize_error(self, error_type: ErrorType) -> ErrorCategory:
        """分类错误"""
        category_mapping = {
            ErrorType.VALIDATION_ERROR: ErrorCategory.CLIENT_ERROR,
            ErrorType.AUTHENTICATION_ERROR: ErrorCategory.CLIENT_ERROR,
            ErrorType.AUTHORIZATION_ERROR: ErrorCategory.CLIENT_ERROR,
            ErrorType.BUSINESS_ERROR: ErrorCategory.SERVER_ERROR,
            ErrorType.SYSTEM_ERROR: ErrorCategory.SYSTEM_ERROR,
            ErrorType.NETWORK_ERROR: ErrorCategory.NETWORK_ERROR,
            ErrorType.TIMEOUT_ERROR: ErrorCategory.NETWORK_ERROR,
            ErrorType.RESOURCE_ERROR: ErrorCategory.SYSTEM_ERROR,
            ErrorType.CONFIGURATION_ERROR: ErrorCategory.SYSTEM_ERROR,
            ErrorType.DEPENDENCY_ERROR: ErrorCategory.SYSTEM_ERROR
        }
        return category_mapping.get(error_type, ErrorCategory.SYSTEM_ERROR)
    
    def _get_error_details(self, error: Exception) -> str:
        """获取错误详细信息"""
        details = []
        
        if hasattr(error, 'args') and error.args:
            details.append(f"参数: {error.args}")
        
        if hasattr(error, 'errno'):
            details.append(f"错误码: {error.errno}")
        
        if hasattr(error, 'strerror'):
            details.append(f"错误描述: {error.strerror}")
        
        return "; ".join(details)
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_level = self._get_log_level(error_info.severity)
        
        log_message = (
            f"错误 [{error_info.context.error_id}]: "
            f"{error_info.error_type.value} - {error_info.message}"
        )
        
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            log_message += f"\n堆栈跟踪: {error_info.stack_trace}"
        
        self.logger.log(log_level, log_message)
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """获取日志级别"""
        level_mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_mapping.get(severity, logging.ERROR)
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> Optional[RecoveryAction]:
        """尝试错误恢复"""
        if error_info.error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_info.error_type]
                return recovery_func(error_info)
            except Exception as e:
                self.logger.error(f"恢复策略执行失败: {e}")
        
        return None
    
    # ==================== 恢复策略实现 ====================
    
    def _handle_network_error(self, error_info: ErrorInfo) -> RecoveryAction:
        """处理网络错误"""
        # 检查熔断器状态
        circuit_key = f"network_{error_info.context.module}"
        if self._is_circuit_open(circuit_key):
            return RecoveryAction(
                action_type="circuit_breaker_open",
                parameters={"circuit_key": circuit_key},
                success=False,
                error_message="熔断器已打开"
            )
        
        # 尝试重连
        return RecoveryAction(
            action_type="retry_connection",
            parameters={"max_retries": 3, "delay": 1},
            success=True
        )
    
    def _handle_timeout_error(self, error_info: ErrorInfo) -> RecoveryAction:
        """处理超时错误"""
        return RecoveryAction(
            action_type="increase_timeout",
            parameters={"timeout_multiplier": 2},
            success=True
        )
    
    def _handle_resource_error(self, error_info: ErrorInfo) -> RecoveryAction:
        """处理资源错误"""
        return RecoveryAction(
            action_type="cleanup_resources",
            parameters={"force_cleanup": True},
            success=True
        )
    
    def _handle_dependency_error(self, error_info: ErrorInfo) -> RecoveryAction:
        """处理依赖错误"""
        return RecoveryAction(
            action_type="fallback_service",
            parameters={"fallback_enabled": True},
            success=True
        )
    
    def _handle_system_error(self, error_info: ErrorInfo) -> RecoveryAction:
        """处理系统错误"""
        return RecoveryAction(
            action_type="restart_service",
            parameters={"graceful_restart": True},
            success=True
        )
    
    def _is_circuit_open(self, circuit_key: str) -> bool:
        """检查熔断器是否打开"""
        if circuit_key not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[circuit_key]
        if circuit["state"] == "open":
            # 检查是否可以尝试关闭
            if time.time() - circuit["last_failure"] > circuit["timeout"]:
                circuit["state"] = "half_open"
                return False
            return True
        
        return False
    
    def update_circuit_breaker(self, circuit_key: str, success: bool):
        """更新熔断器状态"""
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure": 0,
                "timeout": 60
            }
        
        circuit = self.circuit_breakers[circuit_key]
        
        if success:
            circuit["failure_count"] = 0
            circuit["state"] = "closed"
        else:
            circuit["failure_count"] += 1
            circuit["last_failure"] = time.time()
            
            if circuit["failure_count"] >= 5:  # 失败阈值
                circuit["state"] = "open"

# ==================== 重试装饰器 ====================

def retry_on_error(max_retries: int = 3, 
                   delay: float = 1.0, 
                   backoff_multiplier: float = 2.0,
                   error_types: List[ErrorType] = None):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否是可重试的错误类型
                    if error_types:
                        error_type = _classify_exception(e)
                        if error_type not in error_types:
                            raise e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_multiplier ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if error_types:
                        error_type = _classify_exception(e)
                        if error_type not in error_types:
                            raise e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_multiplier ** attempt)
                        time.sleep(wait_time)
                    else:
                        raise e
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _classify_exception(exception: Exception) -> ErrorType:
    """分类异常"""
    exception_mapping = {
        ValueError: ErrorType.VALIDATION_ERROR,
        TypeError: ErrorType.VALIDATION_ERROR,
        KeyError: ErrorType.VALIDATION_ERROR,
        ConnectionError: ErrorType.NETWORK_ERROR,
        TimeoutError: ErrorType.TIMEOUT_ERROR,
        OSError: ErrorType.SYSTEM_ERROR,
        MemoryError: ErrorType.RESOURCE_ERROR,
        PermissionError: ErrorType.AUTHORIZATION_ERROR
    }
    
    for exc_type, error_type in exception_mapping.items():
        if isinstance(exception, exc_type):
            return error_type
    
    return ErrorType.SYSTEM_ERROR

# ==================== 错误监控器 ====================

class ErrorMonitor:
    """错误监控器"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.metrics: Dict[str, Any] = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "errors_by_module": {},
            "recovery_success_rate": 0.0
        }
    
    def record_error(self, error_info: ErrorInfo):
        """记录错误指标"""
        self.metrics["total_errors"] += 1
        
        # 按类型统计
        error_type = error_info.error_type.value
        self.metrics["errors_by_type"][error_type] = \
            self.metrics["errors_by_type"].get(error_type, 0) + 1
        
        # 按严重程度统计
        severity = error_info.severity.value
        self.metrics["errors_by_severity"][severity] = \
            self.metrics["errors_by_severity"].get(severity, 0) + 1
        
        # 按模块统计
        module = error_info.context.module
        self.metrics["errors_by_module"][module] = \
            self.metrics["errors_by_module"].get(module, 0) + 1
    
    def get_error_report(self) -> Dict[str, Any]:
        """获取错误报告"""
        return {
            "metrics": self.metrics,
            "recent_errors": [
                {
                    "error_id": error.context.error_id,
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.context.timestamp,
                    "module": error.context.module
                }
                for error in self.error_handler.error_history[-10:]  # 最近10个错误
            ],
            "circuit_breakers": self.error_handler.circuit_breakers
        }
    
    def check_error_thresholds(self) -> List[str]:
        """检查错误阈值"""
        alerts = []
        
        # 检查严重错误数量
        critical_errors = self.metrics["errors_by_severity"].get("critical", 0)
        if critical_errors > 5:
            alerts.append(f"严重错误数量过多: {critical_errors}")
        
        # 检查模块错误率
        for module, count in self.metrics["errors_by_module"].items():
            if count > 20:  # 阈值可配置
                alerts.append(f"模块 {module} 错误数量过多: {count}")
        
        return alerts

# ==================== 全局错误处理器 ====================

# 创建全局错误处理器实例
error_handler = ErrorHandler()
error_monitor = ErrorMonitor()

def handle_error(error: Exception, 
                error_type: ErrorType,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                module: str = "unknown",
                additional_data: Optional[Dict[str, Any]] = None) -> ErrorInfo:
    """全局错误处理函数"""
    error_info = error_handler.handle_error(error, error_type, severity, module, additional_data)
    error_monitor.record_error(error_info)
    return error_info

# ==================== 测试函数 ====================

def test_error_handling():
    """测试错误处理功能"""
    print("🧪 测试错误处理机制")
    
    # 测试不同类型的错误
    test_errors = [
        (ValueError("参数验证失败"), ErrorType.VALIDATION_ERROR, ErrorSeverity.LOW),
        (ConnectionError("网络连接失败"), ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM),
        (TimeoutError("操作超时"), ErrorType.TIMEOUT_ERROR, ErrorSeverity.HIGH),
        (MemoryError("内存不足"), ErrorType.RESOURCE_ERROR, ErrorSeverity.CRITICAL)
    ]
    
    for error, error_type, severity in test_errors:
        try:
            error_info = handle_error(error, error_type, severity, "test_module")
            print(f"✅ 错误处理成功: {error_info.error_type.value} - {error_info.message}")
        except Exception as e:
            print(f"❌ 错误处理失败: {e}")
    
    # 测试重试装饰器
    @retry_on_error(max_retries=2, delay=0.1, error_types=[ErrorType.NETWORK_ERROR])
    def test_retry_function():
        raise ConnectionError("模拟网络错误")
    
    try:
        test_retry_function()
    except Exception as e:
        print(f"✅ 重试机制测试完成: {e}")
    
    # 获取错误报告
    report = error_monitor.get_error_report()
    print(f"📊 错误报告: 总计 {report['metrics']['total_errors']} 个错误")

if __name__ == "__main__":
    test_error_handling()
