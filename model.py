#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py - SenseVoice模型接口

该模块为项目提供了统一的SenseVoiceSmall模型接口，
根据可用性自动选择使用funasr_torch或funasr_onnx版本。
"""

import os
import logging
import traceback
from typing import Dict, Any, Union, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model")

class SenseVoiceSmall:
    """
    SenseVoice语音识别模型的统一接口类
    根据环境自动选择使用PyTorch或ONNX版本
    """
    
    def __init__(self, model_dir: str = None, batch_size: int = 10, device: str = None, **kwargs):
        """
        初始化SenseVoice模型
        
        参数:
            model_dir: 模型目录或ModelScope模型ID
            batch_size: 批处理大小
            device: 设备类型 (cpu, cuda:0, etc.)
            **kwargs: 其他参数传递给底层模型
        """
        self.model_dir = model_dir or os.environ.get("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
        self.device = device or os.environ.get("SENSEVOICE_DEVICE", "cpu")
        self.batch_size = batch_size
        self.backend = None  # 'torch' 或 'onnx'
        self.model = None
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """
        初始化底层模型实现
        优先尝试PyTorch版本，若不可用则尝试ONNX版本
        """
        # 先尝试PyTorch版本
        try:
            logger.info("尝试加载PyTorch版本模型...")
            from funasr_torch import SenseVoiceSmall as TorchModel
            from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess
            
            self.model = TorchModel(
                model_dir=self.model_dir,
                batch_size=self.batch_size,
                device=self.device,
                **kwargs
            )
            self.backend = "torch"
            self.postprocess = rich_transcription_postprocess
            logger.info(f"成功加载PyTorch版本模型，设备: {self.device}")
            
        except ImportError:
            logger.warning("PyTorch版本不可用，尝试ONNX版本...")
            try:
                # 尝试ONNX版本
                from funasr_onnx import SenseVoiceSmall as OnnxModel
                from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
                
                # ONNX特有参数设置
                onnx_kwargs = kwargs.copy()
                if self.device.startswith("cuda"):
                    onnx_kwargs["quantize"] = onnx_kwargs.get("quantize", True)
                
                self.model = OnnxModel(
                    model_dir=self.model_dir,
                    batch_size=self.batch_size,
                    **onnx_kwargs
                )
                self.backend = "onnx"
                self.postprocess = rich_transcription_postprocess
                logger.info("成功加载ONNX版本模型")
                
            except ImportError:
                logger.error("无法加载任何版本的SenseVoice模型")
                logger.error("请确保已安装funasr_torch或funasr_onnx")
                raise ImportError("无法导入SenseVoice模型，请安装funasr_torch或funasr_onnx")
            except Exception as e:
                logger.error(f"ONNX模型加载失败: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"PyTorch模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def infer(self, audio_file, language: str = "auto", **kwargs) -> Dict[str, Any]:
        """
        对音频文件进行推理，获取识别结果
        
        参数:
            audio_file: 音频文件路径或音频数据
            language: 语言选择 (auto, zh, en)
            **kwargs: 其他参数传递给底层推理函数
            
        返回:
            识别结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        try:
            # 调用底层模型的推理方法
            result = self.model.infer(audio_file, language=language, **kwargs)
            return result
        except Exception as e:
            logger.error(f"推理过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        返回:
            包含模型信息的字典
        """
        info = {
            "backend": self.backend,
            "model_dir": self.model_dir,
            "device": self.device,
            "batch_size": self.batch_size
        }
        
        # 如果底层模型有get_model_info方法，则调用
        if hasattr(self.model, "get_model_info"):
            info.update(self.model.get_model_info())
            
        return info 