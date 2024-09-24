#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块 - 负责模型的初始化、管理和推理
"""
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import onnxruntime as ort
from funasr_onnx import SenseVoiceSmall

import config
from logger import logger, timer, OperationLogger, PerformanceMonitor

class ModelManager:
    """模型管理类"""
    
    _instance: Optional['ModelManager'] = None
    _model = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型管理器"""
        if not self._initialized:
            self._initialized = True
            self._init_model()
    
    @staticmethod
    def verify_gpu() -> Dict[str, Any]:
        """
        验证GPU环境
        
        返回:
            GPU状态信息字典
        """
        start_time = time.time()
        
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用")
            
            # 获取CUDA设备信息
            device_count = torch.cuda.device_count()
            devices = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                }
                for i in range(device_count)
            ]
            
            # 设置当前设备
            torch.cuda.set_device(0)
            
            # 显存测试
            test_tensor = torch.randn(1000, 1000).cuda()
            current_device = test_tensor.device
            del test_tensor
            
            # 记录验证时间
            verify_time = time.time() - start_time
            PerformanceMonitor.record_metric("gpu_verify_time", verify_time)
            
            result = {
                "status": "available",
                "device_count": device_count,
                "devices": devices,
                "current_device": str(current_device)
            }
            
            logger.info(f"GPU环境验证成功: 设备数量={device_count}, 当前设备={current_device}")
            device_info = [f"{dev['id']}:{dev['name']}" for dev in devices]
            logger.debug(f"可用设备: {device_info}")
            
            return result
        except Exception as e:
            logger.critical(f"GPU环境验证失败: {str(e)}")
            return {
                "status": "unavailable",
                "error": str(e)
            }
    
    def _init_model(self) -> None:
        """初始化语音识别模型"""
        try:
            # 验证GPU环境
            gpu_status = self.verify_gpu()
            if gpu_status["status"] != "available":
                raise RuntimeError(f"GPU环境验证失败: {gpu_status.get('error', '未知错误')}")
            
            # 计算内存限制
            device_id = 0  # 使用第一个可用设备
            gpu_mem_limit = int(torch.cuda.get_device_properties(device_id).total_memory * 0.8)
            
            # 配置ORT提供者
            providers = [
                (
                    "CUDAExecutionProvider", 
                    {
                        "device_id": device_id,
                        "gpu_mem_limit": gpu_mem_limit,
                        "arena_extend_strategy": "kSameAsRequested"
                    }
                )
            ]
            
            # 初始化模型
            start_time = time.time()
            
            self._model = SenseVoiceSmall(
                model_dir=config.MODEL_NAME,
                batch_size=config.MODEL_BATCH_SIZE,
                quantize=config.MODEL_QUANTIZE,
                device=config.MODEL_DEVICE,
                providers=providers,
                download_dir=config.MODEL_CACHE_DIR
            )
            
            # 记录初始化时间
            init_time = time.time() - start_time
            PerformanceMonitor.record_metric("model_init_time", init_time)
            
            OperationLogger.log_operation(
                operation="初始化模型",
                details={
                    "model_name": config.MODEL_NAME,
                    "device": config.MODEL_DEVICE,
                    "init_time": init_time
                }
            )
            
            logger.info(f"模型初始化成功: {config.MODEL_NAME}, 耗时={init_time:.2f}秒")
        except Exception as e:
            OperationLogger.log_operation(
                operation="初始化模型",
                status="failure",
                error=e
            )
            logger.critical(f"模型初始化失败: {str(e)}")
            raise
    
    @timer
    def transcribe(self, audio_files: List[str], language: str = "auto", textnorm: str = "withitn") -> List[str]:
        """
        使用模型转写音频文件
        
        参数:
            audio_files: 音频文件路径列表
            language: 语言设置，默认为自动检测
            textnorm: 文本规范化设置
            
        返回:
            转写结果列表
        """
        try:
            start_time = time.time()
            
            # 执行转写
            results = self._model(audio_files, language=language, textnorm=textnorm)
            
            # 记录转写时间
            transcribe_time = time.time() - start_time
            PerformanceMonitor.record_metric("transcribe_time", transcribe_time)
            
            # 记录详细日志
            details = {
                "file_count": len(audio_files),
                "language": language,
                "textnorm": textnorm,
                "processing_time": transcribe_time
            }
            if len(audio_files) == 1:
                details["file"] = audio_files[0]
                
            OperationLogger.log_operation(
                operation="音频转写",
                details=details
            )
            
            return results
        except Exception as e:
            OperationLogger.log_operation(
                operation="音频转写",
                status="failure",
                details={"files": audio_files},
                error=e
            )
            logger.error(f"音频转写失败: {str(e)}")
            raise 