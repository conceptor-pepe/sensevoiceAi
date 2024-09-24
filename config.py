#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块 - 集中管理应用所有配置项
"""
import os
from pathlib import Path

# --- 基础配置 ---
# GPU设备配置
GPU_DEVICE_ID = 5  # 指定使用的GPU设备ID
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_ID)

# 音频处理配置
TARGET_SAMPLE_RATE = 16000  # 目标采样率（Hz）
SUPPORTED_AUDIO_FORMATS = (
    '.wav', '.mp3', '.aac', '.amr', '.flac', '.ogg', '.opus',
    '.m4a', '.webm', '.wma'
)

# --- 模型配置 ---
MODEL_NAME = "iic/SenseVoiceSmall"  # 模型名称
MODEL_BATCH_SIZE = 10  # 批处理大小
MODEL_QUANTIZE = True  # 是否使用量化
MODEL_DEVICE = "cuda"  # 推理设备
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/modelscope/hub")  # 模型缓存目录

# --- API配置 ---
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1  # Worker进程数
API_TITLE = "SenseVoiceSmall ASR API"
API_DESCRIPTION = "支持多种音频格式的语音识别服务"
API_VERSION = "1.3"

# --- 缓存配置 ---
CACHE_ENABLED = True  # 是否启用缓存
CACHE_DIR = Path("/tmp/senseaudio_cache")  # 缓存目录
CACHE_DIR.mkdir(exist_ok=True, parents=True)  # 确保缓存目录存在
CACHE_MAX_SIZE = 1000  # 最大缓存条目数
CACHE_TTL = 24 * 60 * 60  # 缓存过期时间（秒）

# --- 日志配置 ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = Path("logs")  # 日志目录
LOG_DIR.mkdir(exist_ok=True, parents=True)  # 确保日志目录存在
LOG_FILE = LOG_DIR / "senseaudio.log"  # 日志文件路径
LOG_ROTATION = "1 day"  # 日志轮转周期
LOG_RETENTION = "7 days"  # 日志保留时间 