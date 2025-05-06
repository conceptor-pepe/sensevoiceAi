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
import wave
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Generator, AsyncGenerator

import config
from logger import logger
from stats import TimeStats
from model_manager import model_manager
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

# ============== 音频文件处理函数 ==============
def save_temp_audio(content: bytes, filename: Optional[str] = None) -> str:
    """
    保存音频数据到临时文件
    
    参数:
        content: 音频二进制数据
        filename: 可选的文件名
        
    返回:
        str: 临时文件路径
    """
    # 生成唯一文件名
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:8]
    
    if filename:
        # 如果提供了文件名，保留原始扩展名
        base, ext = os.path.splitext(filename)
        if not ext:  # 如果没有扩展名，默认为.wav
            ext = ".wav"
        temp_file = f"{config.TEMP_DIR}/sv_{timestamp}_{unique_id}{ext}"
    else:
        # 默认为WAV格式
        temp_file = f"{config.TEMP_DIR}/sv_{timestamp}_{unique_id}.wav"
        
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
                logger.debug(f"已删除临时文件: {path}")
        except Exception as e:
            logger.warning(f"删除临时文件失败: {path}, 错误: {str(e)}")

def decode_base64_audio(base64_data: str, stats: Optional[TimeStats] = None) -> Optional[bytes]:
    """
    解码Base64编码的音频数据
    
    参数:
        base64_data: Base64编码的音频数据
        stats: 时间统计对象
        
    返回:
        bytes: 解码后的音频数据，如果解码失败则返回None
    """
    try:
        # 移除可能存在的Base64头（如 data:audio/wav;base64,）
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]
            
        # 确保字符串不包含空格或换行符
        base64_data = base64_data.strip().replace(' ', '').replace('\n', '')
        
        # 解码
        audio_data = base64.b64decode(base64_data)
        
        if stats:
            stats.record_step("Base64解码")
            
        return audio_data
    except Exception as e:
        if stats:
            logger.error(f"[{stats.request_id}] Base64解码失败: {str(e)}")
        else:
            logger.error(f"Base64解码失败: {str(e)}")
        return None

# ============== 标签提取函数 ==============
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

# ============== 音频识别处理函数 ==============
def process_single_audio(audio_path: str, language: str = "auto", use_itn: bool = True, 
                        stats: Optional[TimeStats] = None) -> Dict[str, Any]:
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
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "message": f"音频文件不存在: {audio_path}",
            "time_cost": stats.total_time() if stats else 0
        }
    
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
    
    # 提取原始文本
    raw_text = result
    
    # 后处理文本
    if stats:
        stats.record_step("开始后处理")
    processed_text = rich_transcription_postprocess(result)
    
    # 清理文本中的标签
    clean_text = re.sub(config.TAGS_REGEX, "", result, 0, re.MULTILINE)
    
    # 提取标签
    language_tag, emotion_tag, event_tag = extract_tags(result)
    
    if stats:
        stats.record_step("结果提取")
        
    # 构建响应
    response = {
        "success": True,
        "message": "识别成功",
        "text": processed_text,
        "clean_text": clean_text,
        "raw_text": raw_text,
        "language": language_tag,
        "emotion": emotion_tag,
        "event": event_tag,
        "time_cost": stats.total_time() if stats else 0
    }
    
    if stats:
        response["detail_time"] = stats.get_stats()
        
    return response

def process_multiple_audio(audio_paths: List[str], key_list: List[str], language: str = "auto", 
                          use_itn: bool = True, stats: Optional[TimeStats] = None) -> Dict[str, Any]:
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
    # 验证文件路径
    valid_paths = []
    valid_keys = []
    
    for i, path in enumerate(audio_paths):
        if os.path.exists(path):
            valid_paths.append(path)
            valid_keys.append(key_list[i] if i < len(key_list) else f"audio{i}")
        else:
            logger.warning(f"音频文件不存在，已跳过: {path}")
    
    if not valid_paths:
        return {
            "result": [],
            "message": "未找到有效的音频文件",
            "time_cost": stats.total_time() if stats else 0
        }
    
    # 调用模型进行推理
    results = model_manager.transcribe(valid_paths, language=language, use_itn=use_itn, stats=stats)
    
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
        key = valid_keys[i] if i < len(valid_keys) else f"unknown_{i}"
        
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

# ============== 流式处理函数 ==============
async def process_stream_audio(audio_path: str, language: str = "auto", use_itn: bool = True, 
                         chunk_size_sec: int = 3, stats: Optional[TimeStats] = None) -> AsyncGenerator[str, None]:
    """
    流式处理音频文件
    
    参数:
        audio_path: 音频文件路径
        language: 语言代码
        use_itn: 是否使用反向文本归一化
        chunk_size_sec: 每个块的处理时长（秒）
        stats: 时间统计对象
        
    返回:
        异步生成器，生成每个片段的识别结果（JSON字符串）
    """
    # 检查模型是否已加载
    if not model_manager.is_loaded():
        error_result = {
            "success": False,
            "message": "模型未加载",
            "is_final": True,
            "time_cost": stats.total_time() if stats else 0
        }
        yield json.dumps(error_result)
        return
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        error_result = {
            "success": False,
            "message": f"音频文件不存在: {audio_path}",
            "is_final": True,
            "time_cost": stats.total_time() if stats else 0
        }
        yield json.dumps(error_result)
        return
    
    if stats:
        logger.info(f"[{stats.request_id}] 开始流式处理, 文件: {audio_path}, 语言: {language}, use_itn: {use_itn}")
        stats.record_step("流式处理开始")
        
    try:
        # 读取音频文件
        with wave.open(audio_path, 'rb') as wf:
            frame_rate = wf.getframerate()  # 帧率
            n_channels = wf.getnchannels()  # 通道数
            sample_width = wf.getsampwidth()  # 样本宽度
            
            # 计算每个块的帧数
            chunk_frames = int(frame_rate * chunk_size_sec)
            
            # 流式处理上下文
            context = {"text": "", "last_result": ""}
            chunk_id = 0
            
            # 临时文件列表（用于后续清理）
            temp_files = []
            
            # 循环读取音频数据块
            while True:
                # 读取当前数据块
                frames = wf.readframes(chunk_frames)
                if not frames:
                    break
                
                chunk_id += 1
                if stats:
                    stats.record_step(f"处理第{chunk_id}块")
                
                # 保存当前块到临时文件
                chunk_file = f"{audio_path}_chunk_{chunk_id}.wav"
                temp_files.append(chunk_file)
                
                with wave.open(chunk_file, 'wb') as chunk_wf:
                    chunk_wf.setnchannels(n_channels)
                    chunk_wf.setsampwidth(sample_width)
                    chunk_wf.setframerate(frame_rate)
                    chunk_wf.writeframes(frames)
                
                # 进行当前块的推理
                inference_start = time.time()
                results = model_manager.transcribe([chunk_file], language=language, use_itn=use_itn)
                inference_time = time.time() - inference_start
                
                # 获取当前块的转录结果
                if results and len(results) > 0:
                    current_result = results[0]
                    
                    # 后处理转录结果
                    processed_text = rich_transcription_postprocess(current_result)
                    
                    # 合并结果
                    if chunk_id == 1:
                        # 第一个块的结果直接作为当前结果
                        context["text"] = processed_text
                    else:
                        # 合并新的内容到当前结果
                        # 使用智能文本连接（避免重复）
                        if processed_text:
                            if context["text"]:
                                # 简单文本重叠检测
                                # 获取当前文本的后半部分和新文本的前半部分
                                overlap_len = min(len(context["text"]) // 2, len(processed_text) // 2)
                                if overlap_len > 0:
                                    current_end = context["text"][-overlap_len:]
                                    new_start = processed_text[:overlap_len]
                                    
                                    # 寻找可能的重叠点
                                    max_overlap = 0
                                    for i in range(1, overlap_len + 1):
                                        if current_end[-i:] == new_start[:i]:
                                            max_overlap = i
                                    
                                    # 如果存在重叠，则连接文本时避免重复
                                    if max_overlap > 0:
                                        context["text"] += processed_text[max_overlap:]
                                    else:
                                        context["text"] += " " + processed_text
                                else:
                                    context["text"] += " " + processed_text
                            else:
                                context["text"] = processed_text
                    
                    # 提取标签
                    language_tag, emotion_tag, event_tag = extract_tags(current_result)
                    
                    # 准备输出结果
                    result = {
                        "success": True,
                        "message": "部分识别结果",
                        "text": processed_text,  # 当前块的文本
                        "accumulated_text": context["text"],  # 累积的文本
                        "language": language_tag,
                        "emotion": emotion_tag,
                        "event": event_tag,
                        "is_final": False,  # 标记为非最终结果
                        "chunk_id": chunk_id,
                        "time_cost": inference_time
                    }
                    
                    # 记录此块的内容
                    context["last_result"] = current_result
                    
                    # 删除临时文件
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                        temp_files.remove(chunk_file)
                    
                    # 生成当前块的结果
                    yield json.dumps(result)
            
            # 所有块处理完毕，返回最终结果
            final_result = {
                "success": True,
                "message": "识别完成",
                "text": context["text"],
                "accumulated_text": context["text"],  # 保持一致性
                "language": language_tag if 'language_tag' in locals() else None,
                "emotion": emotion_tag if 'emotion_tag' in locals() else None,
                "event": event_tag if 'event_tag' in locals() else None,
                "is_final": True,  # 标记为最终结果
                "chunk_id": chunk_id,
                "time_cost": stats.total_time() if stats else 0
            }
            
            # 记录统计信息
            if stats:
                stats.record_step("流式处理完成")
                final_result["detail_time"] = stats.get_stats()
                logger.info(f"[{stats.request_id}] 流式转录完成，共{chunk_id}个块, 总耗时: {stats.total_time():.4f}秒")
            
            yield json.dumps(final_result)
            
            # 清理剩余的临时文件
            for file_path in temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
    except Exception as e:
        error_msg = f"流式转录失败: {str(e)}"
        logger.error(f"[{stats.request_id if stats else 'stream'}] {error_msg}")
        
        # 返回错误结果
        yield json.dumps({
            "success": False,
            "message": error_msg,
            "text": "",
            "accumulated_text": context["text"] if "context" in locals() and "text" in context else "",
            "is_final": True,
            "time_cost": stats.total_time() if stats else 0
        })
        
        # 清理可能的临时文件
        if 'temp_files' in locals():
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass 