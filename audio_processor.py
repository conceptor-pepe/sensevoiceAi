#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块 - 处理各种格式的音频文件
"""
import os
import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import librosa
import soundfile as sf
import numpy as np

import config
from logger import logger, timer, OperationLogger, PerformanceMonitor

class AudioCache:
    """音频处理结果缓存管理"""
    
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_file = config.CACHE_DIR / "audio_cache.pkl"
    
    @classmethod
    def _load_cache(cls) -> None:
        """从磁盘加载缓存"""
        if not config.CACHE_ENABLED:
            return
        
        if cls._cache_file.exists():
            try:
                with open(cls._cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    cls._cache = cached_data
                    
                # 清除过期缓存
                now = datetime.now()
                expired_keys = []
                
                for key, value in cls._cache.items():
                    if now - value.get('timestamp', now) > timedelta(seconds=config.CACHE_TTL):
                        expired_keys.append(key)
                        
                for key in expired_keys:
                    del cls._cache[key]
                    
                logger.info(f"成功从磁盘加载缓存，有效条目：{len(cls._cache)}，清除过期条目：{len(expired_keys)}")
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
                cls._cache = {}
        else:
            cls._cache = {}
    
    @classmethod
    def _save_cache(cls) -> None:
        """保存缓存到磁盘"""
        if not config.CACHE_ENABLED:
            return
        
        try:
            # 确保缓存大小控制在配置范围内
            if len(cls._cache) > config.CACHE_MAX_SIZE:
                # 按时间戳排序，保留最新的条目
                sorted_cache = sorted(
                    cls._cache.items(),
                    key=lambda x: x[1].get('timestamp', datetime.min),
                    reverse=True
                )
                cls._cache = dict(sorted_cache[:config.CACHE_MAX_SIZE])
            
            with open(cls._cache_file, 'wb') as f:
                pickle.dump(cls._cache, f)
            logger.debug(f"已保存缓存到磁盘，条目数: {len(cls._cache)}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    @classmethod
    def get_cached_result(cls, audio_hash: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的处理结果
        
        参数:
            audio_hash: 音频文件的哈希值
            
        返回:
            缓存的处理结果或None（如果缓存不存在）
        """
        if not config.CACHE_ENABLED:
            return None
        
        # 延迟加载缓存
        if not cls._cache:
            cls._load_cache()
            
        return cls._cache.get(audio_hash)
    
    @classmethod
    def cache_result(cls, audio_hash: str, result: Dict[str, Any]) -> None:
        """
        缓存处理结果
        
        参数:
            audio_hash: 音频文件的哈希值
            result: 处理结果
        """
        if not config.CACHE_ENABLED:
            return
        
        result['timestamp'] = datetime.now()
        cls._cache[audio_hash] = result
        cls._save_cache()

class AudioProcessor:
    """音频处理类"""
    
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
    def process_audio(file_path: str, target_sr: int = config.TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """
        处理音频文件（重采样、格式转换等）
        
        参数:
            file_path: 音频文件路径
            target_sr: 目标采样率
            
        返回:
            处理后的音频数据和采样率的元组
        """
        start_time = time.time()
        
        try:
            # 使用librosa加载音频（自动重采样）
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # 转换为单声道
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
                
            # 记录处理时间指标
            processing_time = time.time() - start_time
            PerformanceMonitor.record_metric("audio_processing_time", processing_time)
            
            # 记录操作日志
            OperationLogger.log_operation(
                operation="处理音频",
                details={
                    "file_path": file_path,
                    "original_sr": sr,
                    "target_sr": target_sr,
                    "duration": len(y) / target_sr,
                    "processing_time": processing_time
                }
            )
            
            logger.info(f"音频处理完成: 原始采样率→{sr}Hz, 目标采样率→{target_sr}Hz, 持续时间→{len(y)/target_sr:.2f}秒")
            return y, target_sr
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
    
    @staticmethod
    def save_processed_audio(audio_data: np.ndarray, sample_rate: int) -> str:
        """
        保存处理后的音频为临时文件
        
        参数:
            audio_data: 处理后的音频数据
            sample_rate: 采样率
            
        返回:
            临时文件路径
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            return tmp.name
            
import time  # 为了timer装饰器中使用 