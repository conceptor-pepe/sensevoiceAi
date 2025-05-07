#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块 - 实现SenseVoiceSmall模型类封装（优化版）
使用funasr-onnx实现，避免modelscope额外依赖
"""
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import io

import torch
import torchaudio
import numpy as np
from funasr_onnx import SenseVoiceSmall as SV_ONNX

class SenseVoiceSmall:
    """SenseVoiceSmall语音识别模型封装类（优化版）"""
    
    # 单例模式实现
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式，确保只有一个模型实例"""
        if cls._instance is None:
            cls._instance = super(SenseVoiceSmall, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_dir: str, model_instance: Any = None):
        """
        初始化SenseVoiceSmall模型
        
        Args:
            model_dir: 模型目录路径
            model_instance: 预加载的模型实例
        """
        if not self._initialized:
            self.model_dir = model_dir
            self.model = model_instance
            self._initialized = True
        
    @classmethod
    def from_pretrained(cls, model: str, device: str = "cuda:0", **kwargs) -> Tuple["SenseVoiceSmall", Dict[str, Any]]:
        """
        从预训练模型创建SenseVoiceSmall实例
        
        Args:
            model: 模型名称或目录
            device: 推理设备
            **kwargs: 额外参数
            
        Returns:
            (模型实例, 推理参数字典)
        """
        # 设置CUDA设备
        if "cuda" in device:
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            # 确保实际使用cuda设备
            use_device = "cuda"  # 这里是关键修改，确保使用 "cuda" 而不是其他值
            
            # 记录CUDA设备配置信息
            print(f"CUDA配置：CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"当前CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA设备数量: {torch.cuda.device_count()}")
                print(f"CUDA当前设备: {torch.cuda.current_device()}")
                print(f"CUDA设备名称: {torch.cuda.get_device_name(device_id)}")
        else:
            use_device = device
            print(f"使用CPU设备: {device}")
            
        # 提取批处理大小和量化设置
        batch_size = kwargs.get("batch_size", 1)
        quantize = kwargs.get("quantize", True)
        
        print(f"创建SenseVoiceSmall模型，目录: {model}, 设备: {use_device}")
        
        # 配置ONNX运行时的GPU设置
        # 计算内存限制
        gpu_mem_limit = None
        providers = None
        
        if use_device == "cuda" and torch.cuda.is_available():
            gpu_mem_limit = int(torch.cuda.get_device_properties(device_id).total_memory * 0.8)
            # 设置CUDA执行提供者
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
            print(f"配置ONNX运行时使用CUDA，内存限制: {gpu_mem_limit}, 设备ID: {device_id}")
            
        model_instance = SV_ONNX(
            model_dir=model,
            batch_size=batch_size,
            quantize=quantize,
            device=use_device,
            providers=providers,  # 传递执行提供者配置
            download_dir=kwargs.get("download_dir", None)
        )
        
        # 返回封装实例和推理参数
        return cls(model_dir=model, model_instance=model_instance), {}
    
    def eval(self):
        """设置模型为评估模式"""
        # ONNX模型无需设置eval模式
        return self
        
    def inference(
        self, 
        data_in: List[torch.Tensor], 
        language: str = "auto", 
        use_itn: bool = True,
        ban_emo_unk: bool = False,
        key: Optional[List[str]] = None, 
        fs: int = 16000, 
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        执行语音识别推理（优化版，减少临时文件操作）
        
        Args:
            data_in: 输入音频张量列表
            language: 语言设置 
            use_itn: 是否使用文本规范化
            ban_emo_unk: 是否禁用情感和未知标记
            key: 音频键名列表
            fs: 采样率
            **kwargs: 额外参数
            
        Returns:
            识别结果列表
        """
        batch_results = []
        temp_files = []
        file_paths = []
        
        try:
            # 设置文本规范化参数
            textnorm = "withitn" if use_itn else "noitn"
            
            # 处理键名
            actual_keys = []
            for i in range(len(data_in)):
                audio_name = key[i] if key and i < len(key) else f"audio_{i}"
                actual_keys.append(audio_name)
            
            # 尝试直接使用GPU处理音频数据
            if torch.cuda.is_available() and hasattr(self.model, 'infer_from_tensor'):
                try:
                    # 移动音频张量到GPU
                    gpu_tensors = [tensor.cuda() for tensor in data_in]
                    
                    # 直接使用张量进行推理
                    results = self.model.infer_from_tensor(gpu_tensors, language=language, textnorm=textnorm)
                    
                    # 构建结果
                    for i, text in enumerate(results):
                        if i < len(actual_keys):
                            batch_results.append({
                                "key": actual_keys[i],
                                "text": text,
                                "lang": language,
                            })
                    
                    return [batch_results]
                except (AttributeError, Exception) as e:
                    print(f"直接张量推理失败，回退到临时文件方式: {e}")
            
            # 将音频张量保存为临时文件（优化版本）
            for i, audio_tensor in enumerate(data_in):
                # 创建临时文件 - 修复：mkstemp()不支持delete参数
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                
                # 添加到临时文件列表
                temp_files.append(temp_path)
                file_paths.append(temp_path)
                
                # 确保张量格式正确
                tensor_to_save = audio_tensor
                if len(audio_tensor.shape) == 1:
                    tensor_to_save = audio_tensor.unsqueeze(0)
                
                # 保存张量到文件
                torchaudio.save(temp_path, tensor_to_save, fs)
            
            # 调用模型进行推理
            results = self.model(file_paths, language=language, textnorm=textnorm)
            
            # 构建结果
            for i, text in enumerate(results):
                if i < len(actual_keys):
                    batch_results.append({
                        "key": actual_keys[i],
                        "text": text,
                        "lang": language,
                    })
                
            return [batch_results]
        except Exception as e:
            import traceback
            print(f"推理失败: {e}")
            print(traceback.format_exc())
            raise
        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    print(f"清理临时文件失败: {temp_path} - {e}")
        
    def __call__(self, audio_file: Union[str, List[str]], **kwargs) -> List[str]:
        """
        调用模型进行推理（兼容旧API）
        
        Args:
            audio_file: 音频文件路径或路径列表
            **kwargs: 推理参数
            
        Returns:
            识别文本列表
        """
        # 直接调用底层模型
        return self.model(audio_file, **kwargs) 