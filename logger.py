#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 日志管理模块
配置和管理应用的日志记录
"""

import os
import logging
import config

# 日志处理器集合
logging_handlers = []

def setup_logger():
    """
    配置应用日志系统
    """
    # 配置文件处理器
    if config.LOG_FILE:
        # 确保日志目录存在
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 添加文件处理器
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logging_handlers.append(file_handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logging_handlers.append(console_handler)

    # 配置根日志器
    logging.basicConfig(
        level=config.get_log_level(),
        format=config.LOG_FORMAT,
        handlers=logging_handlers
    )
    
    # 配置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return logging.getLogger('sensevoice-api')

# 创建应用日志记录器
logger = setup_logger()

def get_logger():
    """获取应用日志记录器"""
    return logger 