#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块 - 处理各种格式的音频文件（优化版）
"""
import os
import hashlib
import tempfile
import io
from typing import Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
import torchaudio

import config
from logger import logger, timer, OperationLogger, PerformanceMonitor

class AudioProcessor:
    """
    音频处理类（优化版）
    移除缓存和不必要的IO操作，提高性能
    """
    
    _instance = None
    
    def __new__(cls):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(AudioProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        初始化音频处理器
        """
        if not self._initialized:
            self._initialized = True
            logger.info("初始化音频处理器")
    
    @staticmethod
    def compute_audio_hash(audio_data: bytes) -> str:
        """
        计算音频数据的哈希值
        
        参数:
            audio_data: 原始音频二进制数据
            
        返回:
            音频数据的MD5哈希值
        """
        return hashlib.md5(audio_data).hexdigest()
    
    @staticmethod
    @timer
    def process_audio_bytes(audio_data: bytes, target_sr: int = config.TARGET_SAMPLE_RATE) -> torch.Tensor:
        """
        直接处理音频二进制数据，无需写入临时文件
        
        参数:
            audio_data: 音频二进制数据
            target_sr: 目标采样率
            
        返回:
            处理后的音频张量
        """
        try:
            # 创建临时文件，但不保存到磁盘
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(audio_data)
                tmp.flush()
                
                try:
                    # 使用torchaudio加载音频
                    waveform, sr = torchaudio.load(tmp_path)
                    
                    # 转换为单声道
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # 如果需要重采样
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)
                    
                    # 记录处理指标
                    duration = waveform.shape[1] / target_sr
                    PerformanceMonitor.record_metric("audio_processing_time", duration)
                    
                    logger.info(f"音频处理完成: 原始采样率→{sr}Hz, 目标采样率→{target_sr}Hz, 持续时间→{duration:.2f}秒")
                    return waveform
                finally:
                    # 确保临时文件被删除
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {str(e)}")
                        
        except Exception as e:
            # 记录操作失败
            OperationLogger.log_operation(
                operation="处理音频",
                status="failure",
                error=e
            )
            logger.error(f"音频处理失败: {str(e)}")
            raise
    
    @staticmethod
    @timer
    def process_audio(file_path: str, target_sr: int = config.TARGET_SAMPLE_RATE) -> torch.Tensor:
        """
        处理音频文件（重采样、格式转换等）
        
        参数:
            file_path: 音频文件路径
            target_sr: 目标采样率
            
        返回:
            处理后的音频张量
        """
        try:
            # 使用torchaudio加载音频（避免librosa IO操作）
            waveform, sr = torchaudio.load(file_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 如果需要重采样
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            # 记录处理时间指标
            duration = waveform.shape[1] / target_sr
            PerformanceMonitor.record_metric("audio_processing_time", duration)
            
            # 记录操作日志
            OperationLogger.log_operation(
                operation="处理音频",
                details={
                    "file_path": file_path,
                    "original_sr": sr,
                    "target_sr": target_sr,
                    "duration": duration
                }
            )
            
            logger.info(f"音频处理完成: 原始采样率→{sr}Hz, 目标采样率→{target_sr}Hz, 持续时间→{duration:.2f}秒")
            return waveform
        except Exception as e:
            # 记录操作失败
            OperationLogger.log_operation(
                operation="处理音频",
                status="failure",
                details={"file_path": file_path},
                error=e
            )
            logger.error(f"音频处理失败: {str(e)}")
            raise

# 预先初始化单例
audio_processor = AudioProcessor() 