#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块 - 集中管理应用所有配置项（优化版）
"""
import os
import socket
from pathlib import Path

# --- 基础路径配置 ---
# 应用根目录（获取当前文件所在目录的父目录）
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 日志目录路径（系统日志目录）
LOG_DIR = Path("/var/log/sensevoice")
# 不在此处创建目录，因为可能需要root权限，将在启动脚本和服务文件中处理

# --- 基础配置 ---
# GPU设备配置 - 从环境变量获取或使用默认值
GPU_DEVICE_ID = int(os.getenv("SENSEVOICE_GPU_ID", 5))  # 默认使用5号GPU设备
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
MODEL_DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:5")  # 从环境变量获取设备，默认使用CUDA:5
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/modelscope/hub")  # 模型缓存目录

# --- API配置 ---
API_HOST = "0.0.0.0"  # API服务监听地址（所有网络接口）
API_PORT = 8000  # API服务端口
API_WORKERS = 1  # Worker进程数
API_TITLE = "SenseVoiceSmall ASR API"  # API服务标题
API_DESCRIPTION = "支持多种音频格式的语音识别服务"  # API服务描述
API_VERSION = "1.4"  # API版本号
HOSTNAME = socket.gethostname()  # 主机名，用于日志记录和服务识别

# --- 性能优化配置 ---
# ONNX运行时设置
ONNX_INTER_OP_THREADS = int(os.getenv("ONNX_INTER_OP_THREADS", 1))  # 内部操作并行线程数
ONNX_INTRA_OP_THREADS = int(os.getenv("ONNX_INTRA_OP_THREADS", 4))  # 运算符内部线程数

# 设置环境变量以限制CPU线程使用，防止占用过多CPU资源
os.environ["OMP_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["MKL_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(ONNX_INTRA_OP_THREADS)

# 强制启用GPU加速
os.environ["ONNXRUNTIME_CUDA_DEVICE_ID"] = str(GPU_DEVICE_ID)  # 设置ONNX使用的CUDA设备ID
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # 启用TensorRT FP16
os.environ["ORT_TENSORRT_ENABLE"] = "1"  # 启用TensorRT
os.environ["TORCH_CUDA_ARCH_LIST"] = "ALL"  # 启用所有CUDA架构
os.environ["ONNXRUNTIME_PROVIDER"] = "CUDAExecutionProvider"  # 强制使用CUDA执行提供者

# --- 日志配置 ---
LOG_LEVEL = "INFO"  # 日志级别
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志格式
LOG_FILE = LOG_DIR / f"senseaudio.log"  # 主日志文件
ERROR_LOG_FILE = LOG_DIR / f"senseaudio_error.log"  # 错误日志文件
ACCESS_LOG_FILE = LOG_DIR / f"senseaudio_access.log"  # 访问日志文件
LOG_ROTATION = "1 day"  # 日志轮转周期（每天轮转）
LOG_RETENTION = "7 days"  # 日志保留时间（保留7天）
LOG_COMPRESSION = "gz"  # 日志压缩格式
MAX_LOG_SIZE = "100MB"  # 单个日志文件最大大小 