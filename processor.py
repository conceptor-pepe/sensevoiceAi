#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 音频处理模块
负责音频文件处理和结果解析
"""

import os
import re
import time
import base64
import json
from typing import List, Dict, Any, Optional, Tuple, Generator, AsyncGenerator

import config
from logger import logger
from stats import TimeStats
from model_manager import model_manager
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

def save_temp_audio(content: bytes, filename: Optional[str] = None) -> str:
    """
    保存音频数据到临时文件
    
    参数:
        content: 音频二进制数据
        filename: 可选的文件名
        
    返回:
        str: 临时文件路径
    """
    timestamp = int(time.time() * 1000)
    if filename:
        temp_file = f"{config.TEMP_DIR}/sensevoice_{timestamp}_{filename}"
    else:
        temp_file = f"{config.TEMP_DIR}/sensevoice_{timestamp}.wav"
        
    # 确保临时目录存在
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    # 写入文件
    with open(temp_file, "wb") as f:
        f.write(content)
        
    return temp_file

def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    清理临时文件
    
    参数:
        file_paths: 要删除的文件路径列表
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {path}, error: {str(e)}")

def decode_base64_audio(base64_data: str, stats: TimeStats = None) -> Optional[bytes]:
    """
    解码Base64编码的音频数据
    
    参数:
        base64_data: Base64编码的音频数据
        stats: 时间统计对象
        
    返回:
        bytes: 解码后的音频数据，如果解码失败则返回None
    """
    try:
        audio_data = base64.b64decode(base64_data)
        if stats:
            stats.record_step("Base64解码")
        return audio_data
    except Exception as e:
        if stats:
            logger.error(f"[{stats.request_id}] Base64解码失败: {str(e)}")
        else:
            logger.error(f"Base64 decoding failed: {str(e)}")
        return None

def extract_tags(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从识别结果中提取语言、情感和事件标签
    
    参数:
        text: 识别结果文本
        
    返回:
        Tuple[Optional[str], Optional[str], Optional[str]]: 语言标签、情感标签和事件标签
    """
    language_tag = None
    emotion_tag = None
    event_tag = None
    
    # 提取语言标签
    for tag in config.LANGUAGE_TAGS:
        if tag in text:
            language_tag = tag.replace("<|", "").replace("|>", "")
            break
            
    # 提取情感标签
    for tag in config.EMOTION_TAGS:
        if tag in text:
            emotion_tag = tag.replace("<|", "").replace("|>", "")
            break
            
    # 提取事件标签
    for tag in config.EVENT_TAGS:
        if tag in text:
            event_tag = tag.replace("<|", "").replace("|>", "")
            break
            
    return language_tag, emotion_tag, event_tag

def process_single_audio(audio_path: str, language: str = "auto", use_itn: bool = True, 
                        stats: TimeStats = None) -> Dict[str, Any]:
    """
    处理单个音频文件
    
    参数:
        audio_path: 音频文件路径
        language: 语言代码
        use_itn: 是否使用反向文本归一化
        stats: 时间统计对象
        
    返回:
        Dict: 处理结果字典
    """
    # 调用模型进行推理
    results = model_manager.transcribe([audio_path], language=language, use_itn=use_itn, stats=stats)
    
    if not results or len(results) == 0:
        return {
            "success": False,
            "message": "模型处理失败，未返回结果",
            "time_cost": stats.total_time() if stats else 0
        }
    
    # 处理结果
    result = results[0]
    
    # 后处理文本
    if stats:
        stats.record_step("开始后处理")
    processed_text = rich_transcription_postprocess(result)
    
    # 提取标签
    language_tag, emotion_tag, event_tag = extract_tags(result)
    
    if stats:
        stats.record_step("结果提取")
        
    # 构建响应
    response = {
        "success": True,
        "message": "识别成功",
        "text": processed_text,
        "language": language_tag,
        "emotion": emotion_tag,
        "event": event_tag,
        "time_cost": stats.total_time() if stats else 0
    }
    
    if stats:
        response["detail_time"] = stats.get_stats()
        
    return response

def process_multiple_audio(audio_paths: List[str], key_list: List[str], language: str = "auto", 
                          use_itn: bool = True, stats: TimeStats = None) -> Dict[str, Any]:
    """
    批量处理多个音频文件
    
    参数:
        audio_paths: 音频文件路径列表
        key_list: 与音频文件对应的键名列表
        language: 语言代码
        use_itn: 是否使用反向文本归一化
        stats: 时间统计对象
        
    返回:
        Dict: 处理结果字典
    """
    # 调用模型进行推理
    results = model_manager.transcribe(audio_paths, language=language, use_itn=use_itn, stats=stats)
    
    if not results or len(results) == 0:
        return {
            "result": [],
            "message": "批量处理未返回结果",
            "time_cost": stats.total_time() if stats else 0
        }
    
    # 处理结果
    output_results = []
    
    if stats:
        stats.record_step("开始后处理")
        
    for i, result in enumerate(results):
        # 获取当前文件对应的key
        key = key_list[i] if i < len(key_list) else f"unknown_{i}"
        
        # 提取标签
        language_tag, emotion_tag, event_tag = extract_tags(result)
        
        # 生成不同处理级别的文本
        raw_text = result
        clean_text = re.sub(config.TAGS_REGEX, "", result, 0, re.MULTILINE)
        processed_text = rich_transcription_postprocess(result)
        
        # 添加到结果列表
        output_results.append({
            "key": key,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "text": processed_text,
            "language": language_tag,
            "emotion": emotion_tag,
            "event": event_tag
        })
    
    if stats:
        stats.record_step("结果提取")
    
    # 构建响应
    response = {
        "result": output_results,
        "time_cost": stats.total_time() if stats else 0
    }
    
    if stats:
        response["detail_time"] = stats.get_stats()
        
    return response

async def process_stream_audio(audio_path: str, language: str = "auto", use_itn: bool = True, 
                         chunk_size_sec: int = 3, stats: TimeStats = None) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式处理音频文件
    
    参数:
        audio_path: 音频文件路径
        language: 语言代码
        use_itn: 是否使用反向文本归一化
        chunk_size_sec: 每个块的处理时长（秒）
        stats: 时间统计对象
        
    返回:
        异步生成器，生成每个片段的识别结果
    """
    if not model_manager.is_loaded():
        error_result = {
            "success": False,
            "message": "模型未加载",
            "is_final": True,
            "time_cost": stats.total_time() if stats else 0
        }
        yield json.dumps(error_result)
        return
    
    if stats:
        logger.info(f"[{stats.request_id}] 开始流式处理, 文件: {audio_path}, 语言: {language}, use_itn: {use_itn}")
        
    try:
        # 调用模型管理器的流式转录函数
        stream_results = model_manager.transcribe_stream(audio_path, language, use_itn, chunk_size_sec, stats)
        
        # 逐个处理结果块
        for result in stream_results:
            # 转换为JSON格式返回
            result_json = json.dumps(result)
            yield result_json
            
            # 如果是最终结果，记录完成信息
            if result.get("is_final", False) and stats:
                stats.log_stats(prefix="流式处理完成，")
                
    except Exception as e:
        error_msg = f"流式处理异常: {str(e)}"
        logger.error(f"[{stats.request_id if stats else 'stream'}] {error_msg}")
        
        # 返回错误信息
        error_result = {
            "success": False,
            "message": error_msg,
            "is_final": True,
            "time_cost": stats.total_time() if stats else 0
        }
        yield json.dumps(error_result) 