#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块 - 提供日志记录和性能统计功能
"""
import time
import logging
import functools
import inspect
import sys
import os
from pathlib import Path
from typing import Callable, Any, Optional, Dict, Union
import json
from datetime import datetime

from loguru import logger
import config

# --- 配置loguru日志系统 ---

# 移除默认处理器
logger.remove()

# 检查是否使用标准错误输出代替日志文件
use_stderr = os.getenv("USE_STDERR", "false").lower() in ("true", "1", "yes")

# 添加主日志文件处理器
if not use_stderr and config.LOG_DIR.exists() and os.access(config.LOG_DIR, os.W_OK):
    try:
        logger.add(
            sink=config.LOG_FILE,
            level=config.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {process}:{thread} | {name}:{function}:{line} - {message}",
            rotation=config.LOG_ROTATION,  # 日志轮转周期
            retention=config.LOG_RETENTION,  # 保留时间
            compression=config.LOG_COMPRESSION,  # 压缩格式
            encoding="utf-8",
            enqueue=True  # 多进程安全
        )
        
        # 添加错误日志文件处理器（仅记录ERROR及以上级别）
        logger.add(
            sink=config.ERROR_LOG_FILE,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {process}:{thread} | {name}:{function}:{line} - {message}",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression=config.LOG_COMPRESSION,
            encoding="utf-8",
            enqueue=True
        )
        
        # 添加访问日志文件处理器（使用专用处理器）
        access_logger = logger.bind(category="access")
        logger.add(
            sink=config.ACCESS_LOG_FILE,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | ACCESS | {message}",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression=config.LOG_COMPRESSION,
            encoding="utf-8",
            enqueue=True,
            filter=lambda record: "category" in record["extra"] and record["extra"]["category"] == "access"
        )
    except Exception as e:
        print(f"无法配置日志文件: {e}，将使用标准错误输出代替")
        use_stderr = True

# 永远添加控制台输出
logger.add(
    sink=sys.stderr,
    level=config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>" 
        if os.getenv("DEBUG", "false").lower() in ("true", "1", "yes") or os.getenv("ENVIRONMENT", "").lower() != "production"
        else "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    colorize=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes") or os.getenv("ENVIRONMENT", "").lower() != "production",
    enqueue=True
)

# 设置异常捕获
logger.configure(
    handlers=[
        {"sink": lambda _: sys.exit(1), "level": "CRITICAL", "diagnose": True}
    ]
)

# 记录系统启动信息
log_mode = "标准输出模式" if use_stderr else "文件日志模式"
logger.info(f"系统启动: 主机={config.HOSTNAME}, PID={os.getpid()}, 版本={config.API_VERSION}, 日志模式={log_mode}")
if use_stderr:
    logger.warning(f"无法使用日志文件，使用标准输出代替，日志目录: {config.LOG_DIR}")

# --- 时间统计装饰器 ---
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
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            logger.exception(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
        finally:
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录执行时间
            status = "成功" if success else "失败"
            logger.info(f"函数 {func.__name__} 执行{status}，耗时: {execution_time:.4f}秒")
            
            # 记录性能指标
            PerformanceMonitor.record_metric(f"{func.__name__}_time", execution_time)
        
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        
        # 执行同步函数
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            logger.exception(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
        finally:
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录执行时间
            status = "成功" if success else "失败"
            logger.info(f"函数 {func.__name__} 执行{status}，耗时: {execution_time:.4f}秒")
            
            # 记录性能指标
            PerformanceMonitor.record_metric(f"{func.__name__}_time", execution_time)
        
        return result
    
    # 根据函数类型返回相应的包装器
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# --- 操作日志记录 ---
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
            "host": config.HOSTNAME,
            "details": details or {}
        }
        
        if error:
            log_entry["error"] = str(error)
            log_entry["error_type"] = error.__class__.__name__
            logger.error(f"操作失败: {operation} - {error}")
            logger.error(f"详细信息: {json.dumps(log_entry, ensure_ascii=False)}")
        else:
            logger.info(f"操作成功: {operation}")
            logger.debug(f"详细信息: {json.dumps(log_entry, ensure_ascii=False)}")

    @staticmethod
    def log_api_access(
        endpoint: str,
        method: str,
        client_ip: str,
        status_code: int,
        processing_time: float,
        request_data: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> None:
        """
        记录API访问日志
        
        参数:
            endpoint: API端点
            method: HTTP方法
            client_ip: 客户端IP
            status_code: HTTP状态码
            processing_time: 处理时间(秒)
            request_data: 请求数据
            error: 错误信息
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "client_ip": client_ip,
            "status_code": status_code,
            "processing_time": processing_time,
            "host": config.HOSTNAME
        }
        
        if request_data:
            # 过滤请求数据中的敏感信息或大对象
            filtered_data = {}
            for key, value in request_data.items():
                if isinstance(value, (str, int, float, bool)) and key not in ("audio", "files"):
                    filtered_data[key] = value
                elif key in ("audio", "files"):
                    filtered_data[key] = "[BINARY_DATA]"
            log_entry["request"] = filtered_data
            
        if error:
            log_entry["error"] = error
        
        # 使用专用访问日志器记录
        json_entry = json.dumps(log_entry, ensure_ascii=False)
        if hasattr(logger, 'bind') and not use_stderr:
            access_logger.info(json_entry)
        else:
            # 如果无法使用专用访问日志，回退到普通日志
            logger.info(f"API访问: {json_entry}")
            
# --- 性能监控 ---
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
    
    @classmethod
    def export_metrics(cls) -> str:
        """导出性能指标为JSON格式"""
        metrics = cls.get_metrics()
        return json.dumps(metrics, ensure_ascii=False, indent=2) 