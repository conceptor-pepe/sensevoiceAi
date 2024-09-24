#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块 - 提供日志记录和性能统计功能
"""
import time
import logging
import functools
import inspect
from pathlib import Path
from typing import Callable, Any, Optional, Dict, Union
import json
from datetime import datetime

from loguru import logger
import config

# 配置loguru日志
logger.remove()  # 移除默认处理器
logger.add(
    sink=config.LOG_FILE,
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    rotation=config.LOG_ROTATION,
    retention=config.LOG_RETENTION,
    encoding="utf-8",
    enqueue=True
)
# 添加控制台输出
logger.add(
    sink=lambda msg: print(msg),
    level=config.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    colorize=True,
    enqueue=True
)

# 时间统计装饰器（支持同步和异步函数）
def timer(func: Callable) -> Callable:
    """
    装饰器: 计算并记录函数执行时间，支持同步和异步函数
    
    参数:
        func: 被装饰的函数
        
    返回:
        装饰后的函数
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        
        # 执行异步函数
        result = await func(*args, **kwargs)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录执行时间
        logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.4f}秒")
        
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        
        # 执行同步函数
        result = func(*args, **kwargs)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录执行时间
        logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.4f}秒")
        
        return result
    
    # 根据函数类型返回相应的包装器
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# 操作日志记录
class OperationLogger:
    """操作日志管理类"""
    
    @staticmethod
    def log_operation(
        operation: str, 
        status: str = "success", 
        details: Optional[Dict] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        记录操作日志
        
        参数:
            operation: 操作名称
            status: 操作状态 (success/failure)
            details: 操作详情
            error: 如果失败，错误信息
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "status": status,
            "details": details or {}
        }
        
        if error:
            log_entry["error"] = str(error)
            logger.error(f"操作失败: {operation} - {error}")
            logger.error(f"详细信息: {json.dumps(log_entry, ensure_ascii=False)}")
        else:
            logger.info(f"操作成功: {operation}")
            logger.debug(f"详细信息: {json.dumps(log_entry, ensure_ascii=False)}")
            
# 性能监控
class PerformanceMonitor:
    """性能监控类"""
    
    _metrics: Dict[str, Dict[str, Union[int, float]]] = {}
    
    @classmethod
    def record_metric(cls, metric_name: str, value: float) -> None:
        """
        记录性能指标
        
        参数:
            metric_name: 指标名称
            value: 指标值
        """
        if metric_name not in cls._metrics:
            cls._metrics[metric_name] = {
                "count": 0,
                "total": 0.0,
                "min": float('inf'),
                "max": float('-inf')
            }
            
        stats = cls._metrics[metric_name]
        stats["count"] += 1
        stats["total"] += value
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        
    @classmethod
    def get_metrics(cls) -> Dict[str, Dict[str, Union[int, float]]]:
        """获取所有性能指标"""
        metrics = {}
        
        for name, stats in cls._metrics.items():
            metrics[name] = {
                **stats,
                "avg": stats["total"] / stats["count"] if stats["count"] > 0 else 0
            }
            
        return metrics
    
    @classmethod
    def reset_metrics(cls) -> None:
        """重置性能指标"""
        cls._metrics = {} 