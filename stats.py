#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 时间统计工具
用于记录和分析API处理各阶段的耗时
"""

import time
import uuid
from logger import logger

class TimeStats:
    """
    时间统计类
    记录处理过程中各阶段的时间统计
    """
    
    def __init__(self, prefix=""):
        """
        初始化时间统计对象
        
        参数:
            prefix: 请求标识前缀，用于生成唯一请求ID
        """
        self.start_time = time.time()
        self.steps = {}
        self.last_step_time = self.start_time
        
        # 生成唯一请求ID
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        self.request_id = f"{prefix}_{timestamp}_{unique_id}" if prefix else f"req_{timestamp}_{unique_id}"
    
    def __enter__(self):
        """
        上下文管理器入口方法，支持with语句
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出方法，支持with语句
        """
        if exc_type:
            logger.error(f"[{self.request_id}] 执行过程发生异常: {exc_val}")
        return False  # 不抑制异常传播
    
    def record_step(self, step_name):
        """
        记录一个处理步骤的时间
        
        参数:
            step_name: 步骤名称
        """
        now = time.time()
        step_time = now - self.last_step_time
        self.steps[step_name] = step_time
        self.last_step_time = now
        
        # 记录步骤耗时
        logger.debug(f"[{self.request_id}] 步骤 '{step_name}' 耗时: {step_time:.4f}秒")
        
        return step_time
    
    def total_time(self):
        """返回总处理时间"""
        return time.time() - self.start_time
    
    def get_stats(self):
        """获取所有时间统计信息"""
        stats = {k: round(v, 4) for k, v in self.steps.items()}
        stats["total"] = round(self.total_time(), 4)
        return stats
    
    def log_stats(self, level="info", prefix=""):
        """
        记录所有统计信息到日志
        
        参数:
            level: 日志级别 (debug, info, warning, error)
            prefix: 日志前缀
        """
        stats = self.get_stats()
        
        if level == "debug":
            logger.debug(f"[{self.request_id}] {prefix}时间统计: {stats}")
        elif level == "warning":
            logger.warning(f"[{self.request_id}] {prefix}时间统计: {stats}")
        elif level == "error":
            logger.error(f"[{self.request_id}] {prefix}时间统计: {stats}")
        else:  # 默认info级别
            logger.info(f"[{self.request_id}] {prefix}时间统计: {stats}")
        
        return stats 