#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SenseVoice API 配置管理模块
负责集中管理所有配置参数和环境变量
"""

import os
import logging

# API服务配置
API_HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")  # 服务监听地址,默认0.0.0.0
API_PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))  # 服务端口,默认8000

# 模型配置
MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")  # 模型目录
GPU_DEVICE = os.getenv("SENSEVOICE_GPU_DEVICE", "5")  # 默认使用GPU 5
BATCH_SIZE = int(os.getenv("SENSEVOICE_BATCH_SIZE", "1"))  # 根据显存调整

# 日志配置
LOG_LEVEL = os.environ.get("SENSEVOICE_LOG_LEVEL", "INFO")  # 日志级别,默认INFO
LOG_FORMAT = os.environ.get("SENSEVOICE_LOG_FORMAT", 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式
# 日志文件路径,默认为空表示仅输出到控制台
# 如需输出到文件,可通过环境变量SENSEVOICE_LOG_FILE设置,例如: /var/log/sensevoice/sensevoice.log
LOG_FILE = os.environ.get("SENSEVOICE_LOG_FILE", "")  

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 临时文件配置
TEMP_DIR = os.environ.get("SENSEVOICE_TEMP_DIR", "/tmp")  # 临时文件存储目录

# 语言支持
SUPPORTED_LANGUAGES = ["auto", "zh", "en", "yue", "ja", "ko"]  # 支持的语言列表

# 标签正则表达式
TAGS_REGEX = r"<\|.*?\|>"  # 用于匹配标签的正则表达式

# 语言标签
LANGUAGE_TAGS = ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]  # 语言标识标签

# 情感标签
EMOTION_TAGS = ["<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", 
                "<|FEARFUL|>", "<|DISGUSTED|>", "<|SURPRISED|>"]  # 情感标识标签

# 事件标签
EVENT_TAGS = ["<|BGM|>", "<|Speech|>", "<|Applause|>", "<|Laughter|>", 
              "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>"]  # 事件标识标签

# 显存限制（根据您的GPU调整）
GPU_MEMORY_LIMIT = 20 * 1024 * 1024 * 1024  # 20GB for RTX 4090

def get_log_level():
    """获取配置的日志级别"""
    return LOG_LEVELS.get(LOG_LEVEL.upper(), logging.INFO)

def print_config():
    """打印当前配置信息"""
    config_info = [
        "================================",
        "SenseVoice API 配置信息",
        f"主机: {API_HOST}",
        f"端口: {API_PORT}", 
        f"模型目录: {MODEL_DIR}",
        f"GPU设备: {GPU_DEVICE}",
        f"批处理大小: {BATCH_SIZE}",
        f"日志级别: {LOG_LEVEL}",
    ]
    
    if LOG_FILE:
        config_info.append(f"日志文件: {LOG_FILE}")
    
    config_info.append("================================")
    
    for line in config_info:
        print(line)
    
    return config_info 