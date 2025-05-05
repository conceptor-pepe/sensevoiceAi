#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 配置管理模块
负责集中管理所有配置参数和环境变量
"""

import os
import logging

# API服务配置
API_HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))

# 模型配置
MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
GPU_DEVICE = os.environ.get("SENSEVOICE_GPU_DEVICE", "0")
BATCH_SIZE = int(os.environ.get("SENSEVOICE_BATCH_SIZE", "1"))

# 日志配置
LOG_LEVEL = os.environ.get("SENSEVOICE_LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("SENSEVOICE_LOG_FORMAT", 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
TEMP_DIR = os.environ.get("SENSEVOICE_TEMP_DIR", "/tmp")

# 语言支持
SUPPORTED_LANGUAGES = ["auto", "zh", "en", "yue", "ja", "ko"]

# 标签正则表达式
TAGS_REGEX = r"<\|.*?\|>"

# 语言标签
LANGUAGE_TAGS = ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]

# 情感标签
EMOTION_TAGS = ["<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", 
                "<|FEARFUL|>", "<|DISGUSTED|>", "<|SURPRISED|>"]

# 事件标签
EVENT_TAGS = ["<|BGM|>", "<|Speech|>", "<|Applause|>", "<|Laughter|>", 
              "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>"]

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