#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块 - 实现SenseVoiceSmall模型类封装
使用funasr-onnx实现，避免modelscope额外依赖
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from funasr_onnx import SenseVoiceSmall as SV_ONNX

class SenseVoiceSmall:
    """SenseVoiceSmall语音识别模型封装类"""
    
    def __init__(self, model_dir: str, model_instance: Any = None):
        """
        初始化SenseVoiceSmall模型
        
        Args:
            model_dir: 模型目录路径
            model_instance: 预加载的模型实例
        """
        self.model_dir = model_dir
        self.model = model_instance
        
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
            use_device = "cuda"
        else:
            use_device = device
            
        # 提取批处理大小和量化设置
        batch_size = kwargs.get("batch_size", 1)
        quantize = kwargs.get("quantize", True)
        
        # 创建模型实例
        print(f"创建SenseVoiceSmall模型，目录: {model}, 设备: {use_device}")
        model_instance = SV_ONNX(
            model_dir=model,
            batch_size=batch_size,
            quantize=quantize,
            device=use_device,
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
        执行语音识别推理
        
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
        
        # 准备音频文件
        audio_paths = []
        temp_files = []
        
        try:
            # 将张量保存为临时WAV文件
            import tempfile
            for i, audio_tensor in enumerate(data_in):
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                # 保存音频
                torchaudio.save(temp_path, audio_tensor.unsqueeze(0), fs)
                
                # 添加到列表
                audio_paths.append(temp_path)
                temp_files.append(temp_path)
            
            # 使用funasr-onnx模型进行推理
            textnorm = "withitn" if use_itn else "noitn"
            results = self.model(audio_paths, language=language, textnorm=textnorm)
            
            # 格式化结果
            for i, text in enumerate(results):
                audio_name = key[i] if key and i < len(key) else f"audio_{i}"
                batch_results.append({
                    "key": audio_name,
                    "text": text,
                    "lang": language,
                })
        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    print(f"清理临时文件失败: {temp_path} - {e}")
                
        return [batch_results]
        
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