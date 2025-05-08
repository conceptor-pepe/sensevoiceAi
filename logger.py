#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SenseVoice 日志模块
负责配置和管理应用程序的日志功能
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
import config
from datetime import datetime

# 全局日志记录器
logger = None

def ensure_log_dir():
    """
    确保日志目录存在，如果不存在则创建
    
    Returns:
        bool: 目录是否创建或已存在的成功标志
    """
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"无法创建日志目录 {config.LOG_DIR}: {str(e)}")
        return False

def setup_logger():
    """
    设置并返回全局日志记录器
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    global logger
    
    # 如果已经设置过日志记录器，直接返回
    if logger is not None:
        return logger
    
    # 确保日志目录存在
    ensure_log_dir()
    
    # 创建日志记录器
    logger = logging.getLogger("sensevoice")
    
    # 设置日志级别
    log_level = getattr(logging, config.LOG_LEVEL)
    logger.setLevel(log_level)
    
    # 清空现有的处理器（防止重复）
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    log_file = os.path.join(config.LOG_DIR, 
                           datetime.now().strftime(config.LOG_FILENAME_FORMAT))
    
    # 使用 TimedRotatingFileHandler 实现日志滚动
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',     # 每天午夜滚动日志
        interval=1,          # 每1个单位（这里是天）滚动一次
        backupCount=30,      # 保留30个备份文件
        encoding='utf-8'     # 使用UTF-8编码
    )
    
    # 设置日志格式
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 初始化全局日志记录器
logger = setup_logger() 