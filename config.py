#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块 - 集中管理应用所有配置项（优化版）
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
API_VERSION = "1.4"  # 更新版本号

# --- 缓存配置 ---
# 禁用文件缓存以避免IO操作
CACHE_ENABLED = False  # 禁用缓存
CACHE_DIR = Path("/tmp/senseaudio_cache")  # 缓存目录（即使禁用也保留配置项）
CACHE_DIR.mkdir(exist_ok=True, parents=True)  # 确保缓存目录存在
CACHE_MAX_SIZE = 0  # 最大缓存条目数（已禁用）
CACHE_TTL = 0  # 缓存过期时间（已禁用）

# --- 性能优化配置 ---
# ONNX运行时设置
ONNX_INTER_OP_THREADS = 1  # 内部操作并行线程数
ONNX_INTRA_OP_THREADS = 4  # 运算符内部线程数
# 设置环境变量
os.environ["OMP_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["MKL_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # 启用TensorRT FP16

# --- 日志配置 ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)  # 确保日志目录存在
LOG_FILE = LOG_DIR / "senseaudio.log"  # 日志文件路径
LOG_ROTATION = "1 day"  # 日志轮转周期
LOG_RETENTION = "7 days"  # 日志保留时间 