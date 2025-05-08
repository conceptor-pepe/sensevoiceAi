#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存监控工具
提供内存使用情况监控和管理功能
"""

import os
import psutil
import time
from logger import logger
import config

class MemoryMonitor:
    """内存监控器类，负责监控系统内存使用情况，防止内存溢出"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        获取当前内存使用率
        
        Returns:
            float: 当前内存使用率(0.0-1.0)
        """
        # 返回当前系统内存使用率(0.0-1.0)
        return psutil.virtual_memory().percent / 100.0
    
    @staticmethod
    def is_memory_available() -> bool:
        """
        检查是否有足够内存可用
        
        Returns:
            bool: 如果内存使用率低于阈值，返回True；否则返回False
        """
        # 获取当前内存使用率
        memory_usage = MemoryMonitor.get_memory_usage()
        # 检查是否超过阈值
        is_available = memory_usage < config.MEMORY_THRESHOLD
        
        # 如果内存不可用，记录警告
        if not is_available:
            logger.warning(f"内存使用率达到 {memory_usage:.2%}，超过阈值 {config.MEMORY_THRESHOLD:.2%}")
        
        return is_available
    
    @staticmethod
    def log_memory_status():
        """记录当前进程和系统的内存使用情况"""
        # 当前进程
        process = psutil.Process(os.getpid())
        # 进程内存使用(MB)
        process_memory = process.memory_info().rss / (1024 * 1024)
        # 系统内存使用率
        system_memory_percent = psutil.virtual_memory().percent
        # 可用内存(MB)
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        logger.info(f"内存状态 - 进程: {process_memory:.2f}MB, 系统使用率: {system_memory_percent:.2f}%, 可用: {available_memory:.2f}MB")
    
    @staticmethod
    def wait_for_memory(check_interval=1.0, max_wait_time=60.0):
        """
        等待直到有足够的内存可用或达到最大等待时间
        
        Args:
            check_interval: 检查间隔(秒)
            max_wait_time: 最大等待时间(秒)
            
        Returns:
            bool: 如果成功获取到足够内存返回True；如果超时返回False
        """
        start_time = time.time()
        while not MemoryMonitor.is_memory_available():
            # 检查是否超时
            if time.time() - start_time > max_wait_time:
                logger.warning(f"等待内存超时，当前内存使用率: {MemoryMonitor.get_memory_usage():.2%}")
                return False
                
            # 记录等待信息
            logger.info(f"等待内存释放，当前使用率: {MemoryMonitor.get_memory_usage():.2%}")
            # 短暂等待
            time.sleep(check_interval)
            
        return True 