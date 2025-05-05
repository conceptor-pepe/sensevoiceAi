#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 模型管理模块
负责模型的加载、初始化和管理
"""

import os
import time
from typing import Optional
import config
from logger import logger
from stats import TimeStats

# 导入SenseVoice Small模型
from funasr_onnx import SenseVoiceSmall

class ModelManager:
    """
    模型管理器类
    负责模型的加载和提供访问接口
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """仅在首次创建实例时初始化"""
        if self._model is None:
            # 设置CUDA设备
            os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_DEVICE
    
    def load_model(self) -> bool:
        """
        加载模型
        
        返回:
            bool: 加载是否成功
        """
        try:
            stats = TimeStats(prefix="model_load")
            logger.info(f"[{stats.request_id}] 正在加载SenseVoice Small模型，模型目录: {config.MODEL_DIR}, 使用GPU: {config.GPU_DEVICE}")
            
            # 加载模型
            self._model = SenseVoiceSmall(
                config.MODEL_DIR, 
                batch_size=config.BATCH_SIZE, 
                quantize=True
            )
            
            # 记录加载时间
            load_time = stats.total_time()
            logger.info(f"[{stats.request_id}] 模型加载成功，耗时: {load_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def get_model(self) -> Optional[SenseVoiceSmall]:
        """
        获取模型实例
        
        返回:
            SenseVoiceSmall: 模型实例，如果未加载则返回None
        """
        return self._model
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        返回:
            bool: 模型是否已加载
        """
        return self._model is not None
    
    def transcribe(self, audio_paths, language="auto", use_itn=True, stats=None):
        """
        转录音频
        
        参数:
            audio_paths: 音频文件路径列表
            language: 语言代码
            use_itn: 是否使用反向文本归一化
            stats: 时间统计对象
            
        返回:
            转录结果列表
        """
        if not self.is_loaded():
            logger.error("模型未加载，无法进行转录")
            return None
        
        if stats:
            logger.info(f"[{stats.request_id}] 开始转录，音频数量: {len(audio_paths)}, 语言: {language}, 使用ITN: {use_itn}")
            
        # 进行推理
        try:
            inference_start = time.time()
            results = self._model(audio_paths, language=language, use_itn=use_itn)
            inference_time = time.time() - inference_start
            
            if stats:
                stats.record_step("模型推理")
                avg_time = inference_time / len(audio_paths) if audio_paths else 0
                logger.info(f"[{stats.request_id}] 转录完成，耗时: {inference_time:.4f}秒，平均每文件: {avg_time:.4f}秒")
            
            return results
            
        except Exception as e:
            if stats:
                logger.error(f"[{stats.request_id}] 转录失败: {str(e)}")
            else:
                logger.error(f"转录失败: {str(e)}")
            return None

# 创建全局模型管理器实例
model_manager = ModelManager() 