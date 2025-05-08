#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SenseVoice API 配置文件
包含所有可配置的参数和设置
"""

import os
from datetime import datetime
from pathlib import Path

# --- 服务配置 ---
# 服务器配置
HOST = "0.0.0.0"  # 服务监听地址，0.0.0.0表示所有网络接口
PORT = 8000  # 服务监听端口
DEBUG = False  # 是否开启调试模式

# --- GPU/设备配置 ---
# DEVICE_ID: 指定使用的计算设备ID，例如GPU的编号
DEVICE_ID = 5
# MODEL_WORKERS: 为GPU推理任务配置的工作线程数，通常为1以避免GPU争用
MODEL_WORKERS = 1

# --- 并发和内存管理配置 ---
# MAX_CONCURRENT_REQUESTS: 最大并发请求数量，超过此数量的请求将排队等待
MAX_CONCURRENT_REQUESTS = 10
# 并发控制的信号量大小
REQUEST_SEMAPHORE_SIZE = 10
# 每个请求最大允许的文件数
MAX_FILES_PER_REQUEST = 50
# 单个音频文件的最大大小(字节)，超过此大小将被拒绝或分片处理
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
# 大文件处理的分片大小(秒)
LARGE_FILE_CHUNK_SIZE_SEC = 60
# 内存使用率阈值，超过此阈值将开始拒绝新请求
MEMORY_THRESHOLD = 0.90  # 90%内存使用率

# --- 模型配置 ---
# 模型目录路径
MODEL_DIR = "iic/SenseVoiceSmall"
# 模型推理时使用的批处理大小
BATCH_SIZE = 16
# 是否使用量化模型。量化模型体积小，速度快，但精度可能略有下降。
QUANTIZE = False

# --- 日志配置 ---
# 日志文件存储目录
LOG_DIR = "/var/log/sensevoice"
# 日志级别 (可选: DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"
# 日志文件命名格式
LOG_FILENAME_FORMAT = f"sensevoice_%Y%m%d.log"
# 当前日志文件路径
LOG_FILE = os.path.join(LOG_DIR, datetime.now().strftime(LOG_FILENAME_FORMAT))
# 日志消息格式
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- 测试配置 ---
# 测试文件目录
TEST_DIR = "test_files"

# --- 其他配置 ---
# 系统基本路径
BASE_DIR = Path(__file__).resolve().parent

# 尝试从本地配置文件加载，覆盖默认配置
try:
    from local_config import *
    print(f"已加载本地配置文件: local_config.py")
except ImportError:
    print(f"未找到本地配置文件，使用默认配置。如需自定义，请创建 local_config.py 文件。") 