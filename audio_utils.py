#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频处理工具
提供音频文件处理、分析和分片相关功能
"""

import librosa
import numpy as np
import io
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import os
from pydub import AudioSegment
import soundfile as sf
from logger import logger
import config

class AudioProcessor:
    """音频处理器类，提供音频文件处理和分析功能"""
    
    @staticmethod
    def get_audio_duration(audio_data: bytes) -> float:
        """
        获取音频文件的时长
        
        Args:
            audio_data: 音频文件的二进制数据
            
        Returns:
            float: 音频时长(秒)
        """
        try:
            # 创建临时文件保存音频数据
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)
            
            # 加载音频并获取时长
            y, sr = librosa.load(temp_filename, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 删除临时文件
            os.unlink(temp_filename)
            
            return duration
        except Exception as e:
            logger.error(f"获取音频时长失败: {str(e)}")
            # 如果无法获取时长，返回0
            return 0.0
    
    @staticmethod
    def is_large_file(audio_data: bytes) -> bool:
        """
        检查是否是大文件
        
        Args:
            audio_data: 音频文件的二进制数据
            
        Returns:
            bool: 如果文件大小超过阈值，返回True；否则返回False
        """
        # 检查文件字节大小
        if len(audio_data) > config.MAX_FILE_SIZE_BYTES:
            return True
            
        # 检查音频时长
        duration = AudioProcessor.get_audio_duration(audio_data)
        # 如果时长大于2分钟，也考虑为大文件
        if duration > 120:  # 2分钟
            logger.info(f"检测到大音频文件，时长: {duration:.2f}秒")
            return True
            
        return False
    
    @staticmethod
    def split_audio(audio_data: bytes, chunk_size_sec: float = None) -> List[Tuple[bytes, float, float]]:
        """
        将音频文件分片
        
        Args:
            audio_data: 音频文件的二进制数据
            chunk_size_sec: 每个分片的时长(秒)，默认使用配置中的值
            
        Returns:
            List[Tuple[bytes, float, float]]: 分片列表，每个元素为(分片数据, 开始时间, 结束时间)
        """
        if chunk_size_sec is None:
            chunk_size_sec = config.LARGE_FILE_CHUNK_SIZE_SEC
            
        try:
            # 创建临时文件保存音频数据
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)
            
            # 使用pydub加载音频
            audio = AudioSegment.from_file(temp_filename)
            # 总时长(毫秒)
            total_duration = len(audio)
            # 分片大小(毫秒)
            chunk_size_ms = int(chunk_size_sec * 1000)
            
            chunks = []
            
            # 按时间分片
            for start_ms in range(0, total_duration, chunk_size_ms):
                end_ms = min(start_ms + chunk_size_ms, total_duration)
                chunk = audio[start_ms:end_ms]
                
                # 将分片保存为临时wav文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as chunk_file:
                    chunk_filename = chunk_file.name
                    chunk.export(chunk_filename, format="wav")
                    
                    # 读取分片文件数据
                    with open(chunk_filename, 'rb') as f:
                        chunk_data = f.read()
                    
                    # 将分片数据、开始时间和结束时间添加到列表
                    start_sec = start_ms / 1000.0
                    end_sec = end_ms / 1000.0
                    chunks.append((chunk_data, start_sec, end_sec))
                    
                    # 删除临时分片文件
                    os.unlink(chunk_filename)
            
            # 删除原始临时文件
            os.unlink(temp_filename)
            
            logger.info(f"音频文件已分成 {len(chunks)} 个分片")
            return chunks
        except Exception as e:
            logger.error(f"分割音频文件失败: {str(e)}")
            # 如果分片失败，返回原始音频作为单个分片
            return [(audio_data, 0.0, AudioProcessor.get_audio_duration(audio_data))]
    
    @staticmethod
    def merge_transcriptions(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并多个分片的转写结果
        
        Args:
            chunk_results: 各个分片的转写结果列表
            
        Returns:
            Dict[str, Any]: 合并后的转写结果
        """
        # 如果只有一个分片，直接返回结果
        if len(chunk_results) == 1:
            return chunk_results[0]
            
        # 初始化合并结果
        merged_result = {
            "processed_text": "",
            "raw_text": "",
            "process_time": 0.0
        }
        
        # 合并文本和处理时间
        for chunk in chunk_results:
            processed_text = chunk.get("processed_text", "")
            raw_text = chunk.get("raw_text", "")
            process_time = chunk.get("process_time", 0.0)
            
            # 添加空格连接文本(如果两段文本都不为空)
            if merged_result["processed_text"] and processed_text:
                merged_result["processed_text"] += " "
            merged_result["processed_text"] += processed_text
            
            if merged_result["raw_text"] and raw_text:
                merged_result["raw_text"] += " "
            merged_result["raw_text"] += raw_text
            
            # 累加处理时间
            merged_result["process_time"] += process_time
        
        return merged_result 