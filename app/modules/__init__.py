"""
功能模块基类

所有模块必须实现统一接口，便于插拔
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModuleResult:
    """模块处理结果的标准格式"""
    module_name: str              # 模块名称
    success: bool                 # 是否成功
    data: Optional[Dict] = None   # 结果数据
    error: Optional[str] = None   # 错误信息
    metadata: Optional[Dict] = None  # 元数据（如耗时、置信度等）


class BaseModule(ABC):
    """模块基类"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.enabled = True
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> ModuleResult:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据字典
        Returns:
            ModuleResult: 处理结果
        """
        pass
    
    def enable(self):
        """启用模块"""
        self.enabled = True
    
    def disable(self):
        """禁用模块"""
        self.enabled = False