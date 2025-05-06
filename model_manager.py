#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 模型管理模块
负责模型的加载、初始化和管理
"""

import os
import time
import subprocess
import numpy as np
from typing import Optional, Generator, List, Dict, Any, Tuple
from collections import deque
from threading import Lock
import torch

import config
from logger import logger
from stats import TimeStats

# 导入SenseVoice Small模型
from funasr_onnx import SenseVoiceSmall

class ModelManager:
    """
    模型管理器类 - GPU优化版
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            # 强制设置GPU环境变量（立即生效）
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_DEVICE)
            os.environ["ORT_CUDA_DEVICE"] = "0"  # 重要修改：设置为0，因为CUDA_VISIBLE_DEVICES已指定了实际设备
            os.environ["OMP_NUM_THREADS"] = "1"  # 避免CPU线程竞争
            
            logger.info(f"强制设置GPU环境: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # 初始化流式处理相关属性
            self._ws_lock = Lock()
            self._sample_rate = 16000
            self._accumulated_results = {}
            
            # 立即检查GPU状态
            self._check_gpu_availability()

    def _check_gpu_availability(self):
        """增强版GPU检查"""
        try:
            # 1. 检查nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '-i', config.GPU_DEVICE, '--query-gpu=name,memory.total', '--format=csv'],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                gpu_info = result.stdout.decode('utf-8').strip()
                logger.info(f"GPU {config.GPU_DEVICE} 信息:\n{gpu_info}")
            else:
                logger.error(f"GPU检查失败: {result.stderr.decode('utf-8')}")

            # 2. 检查ONNX Runtime GPU支持
            import onnxruntime as ort
            providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime可用Providers: {providers}")
            
            if 'CUDAExecutionProvider' not in providers:
                raise RuntimeError("ONNX Runtime未检测到CUDA支持")
                
            # 3. 确认GPU可用性
            # 不再尝试加载空模型，而是简单地检查环境变量和提供程序
            cuda_device_id = int(config.GPU_DEVICE)
            logger.info(f"GPU环境检查: 使用GPU设备ID {cuda_device_id}")
            
            # 强制设置GPU环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_DEVICE)
            os.environ["ORT_CUDA_DEVICE"] = "0"  # 由于设置了CUDA_VISIBLE_DEVICES，在ONNX中使用0号设备
            
            # 验证环境变量
            logger.info(f"已设置环境变量 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            logger.info(f"已设置环境变量 ORT_CUDA_DEVICE={os.environ.get('ORT_CUDA_DEVICE')}")
            
            # 确认CUDA可用
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
                logger.info(f"PyTorch检测到 {device_count} 个GPU设备，当前设备: {device_name}")
            else:
                logger.warning("PyTorch未检测到可用的CUDA设备")
            
            logger.info("GPU检查完成，环境配置正确")
            
        except Exception as e:
            logger.error(f"GPU检查严重错误: {str(e)}")
            raise
    def load_model(self) -> bool:
        """
        优化版模型加载函数
        支持自动重试和错误恢复
        """
        stats = TimeStats(prefix="model_load")
        
        try:
            # 重新强制设置GPU环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_DEVICE)
            os.environ["ORT_CUDA_DEVICE"] = "0"  # 重要：必须是0，因为CUDA_VISIBLE_DEVICES已经筛选了实际GPU
            logger.info(f"[{stats.request_id}] 设置环境变量: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, ORT_CUDA_DEVICE={os.environ['ORT_CUDA_DEVICE']}")

            # 确认ONNX Runtime的GPU支持
            import onnxruntime as ort
            ort_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in ort_providers:
                logger.warning(f"[{stats.request_id}] ONNX Runtime不支持CUDA，将使用CPU推理")
            
            logger.info(f"[{stats.request_id}] ONNX Runtime提供者: {ort_providers}")
            
            # 安全的GPU Provider配置
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,  # 必须为0，因为CUDA_VISIBLE_DEVICES已筛选
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'do_copy_in_default_stream': True
                }),
                'CPUExecutionProvider'  # 恢复CPU Provider作为备选
            ]
            
            # 最大尝试次数
            max_attempts = 3
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                attempt += 1
                logger.info(f"[{stats.request_id}] 尝试加载模型 (第{attempt}次尝试)")
                
                try:
                    # 尝试加载模型
                    if 'CUDAExecutionProvider' in ort_providers:
                        # 使用GPU提供者
                        self._model = SenseVoiceSmall(
                            config.MODEL_DIR,
                            batch_size=config.BATCH_SIZE,
                            quantize=True,
                            providers=providers,
                            device='cuda'
                        )
                        logger.info(f"[{stats.request_id}] 成功使用GPU提供者加载模型")
                    else:
                        # 回退到CPU
                        self._model = SenseVoiceSmall(
                            config.MODEL_DIR,
                            batch_size=config.BATCH_SIZE,
                            quantize=True,
                            providers=['CPUExecutionProvider'],
                            device='cpu'
                        )
                        logger.info(f"[{stats.request_id}] 成功使用CPU提供者加载模型")
                    
                    success = True
                except Exception as e:
                    logger.warning(f"[{stats.request_id}] 模型加载失败 (尝试 {attempt}/{max_attempts}): {str(e)}")
                    
                    if attempt == max_attempts:
                        # 最后一次尝试，尝试不指定providers
                        try:
                            logger.info(f"[{stats.request_id}] 最后尝试: 使用默认提供者")
                            self._model = SenseVoiceSmall(
                                config.MODEL_DIR,
                                batch_size=config.BATCH_SIZE,
                                quantize=True
                            )
                            logger.info(f"[{stats.request_id}] 成功使用默认提供者加载模型")
                            success = True
                        except Exception as final_e:
                            logger.error(f"[{stats.request_id}] 所有加载尝试均失败: {str(final_e)}")
                            raise
                    
                    # 短暂等待后重试
                    if not success and attempt < max_attempts:
                        time.sleep(2)
            
            stats.record_step("模型加载完成")
            logger.info(f"[{stats.request_id}] 模型加载完成，使用时间: {stats.total_time():.4f}秒")
            
            # 验证模型是否可用
            try:
                # 创建一个小的测试输入
                test_audio_data = os.path.join(os.path.dirname(__file__), "test_data", "test.wav")
                
                # 如果测试文件不存在，则创建一个简单的测试WAV文件
                if not os.path.exists(test_audio_data):
                    os.makedirs(os.path.dirname(test_audio_data), exist_ok=True)
                    import numpy as np
                    import wave
                    
                    # 创建一个持续1秒的静音WAV文件
                    sample_rate = 16000
                    duration = 1  # 秒
                    silence = np.zeros(sample_rate * duration, dtype=np.int16)
                    
                    with wave.open(test_audio_data, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(silence.tobytes())
                
                # 执行一次测试推理
                logger.info(f"[{stats.request_id}] 执行模型验证测试...")
                test_start = time.time()
                results = self._model([test_audio_data])
                test_duration = time.time() - test_start
                
                logger.info(f"[{stats.request_id}] 模型验证成功，推理时间: {test_duration:.4f}秒")
                return True
            except Exception as e:
                logger.error(f"[{stats.request_id}] 模型验证失败: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"[{stats.request_id}] 模型加载失败: {str(e)}")
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
            logger.error("模型未加载，无法转录")
            return None
        
        if stats:
            logger.info(f"[{stats.request_id}] 开始转录, 音频数量: {len(audio_paths)}, 语言: {language}, 使用ITN: {use_itn}")
            
        # 进行推理
        try:
            inference_start = time.time()
            results = self._model(audio_paths, language=language, use_itn=use_itn)
            inference_time = time.time() - inference_start
            
            if stats:
                stats.record_step("模型推理")
                avg_time = inference_time / len(audio_paths) if audio_paths else 0
                logger.info(f"[{stats.request_id}] 转录成功，总耗时: {inference_time:.4f}s, 平均每个文件: {avg_time:.4f}s")
            
            return results
            
        except Exception as e:
            if stats:
                logger.error(f"[{stats.request_id}] 转录失败: {str(e)}")
            else:
                logger.error(f"转录失败: {str(e)}")
            return None
    
    def transcribe_stream(self, audio_path: str, language="auto", use_itn=True, 
                          chunk_size_sec=3, stats=None) -> Generator[Dict[str, Any], None, None]:
        """
        流式转录音频
        
        参数:
            audio_path: 音频文件路径
            language: 语言代码
            use_itn: 是否使用反向文本归一化
            chunk_size_sec: 每个音频块的长度（秒）
            stats: 时间统计对象
            
        返回:
            生成器，生成每个片段的转录结果
        """
        import numpy as np
        import wave
        import json
        from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
        
        if not self.is_loaded():
            logger.error("模型未加载，无法进行流式转录")
            return
        
        if stats:
            logger.info(f"[{stats.request_id}] 开始流式转录, 文件: {audio_path}, 语言: {language}, "
                       f"分块大小: {chunk_size_sec}秒, ITN: {use_itn}")
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
                    with wave.open(chunk_file, 'wb') as chunk_wf:
                        chunk_wf.setnchannels(n_channels)
                        chunk_wf.setsampwidth(sample_width)
                        chunk_wf.setframerate(frame_rate)
                        chunk_wf.writeframes(frames)
                    
                    # 进行当前块的推理
                    inference_start = time.time()
                    results = self._model([chunk_file], language=language, use_itn=use_itn)
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
                            # 这里采用简单的文本累加策略
                            # 实际项目中可能需要更复杂的文本合并逻辑
                            context["text"] += " " + processed_text
                        
                        # 提取标签
                        from processor import extract_tags
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
                        
                        # 生成当前块的结果
                        yield result
                
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
                
                yield final_result
                
        except Exception as e:
            error_msg = f"流式转录失败: {str(e)}"
            logger.error(f"[{stats.request_id if stats else 'stream'}] {error_msg}")
            
            # 返回错误结果
            yield {
                "success": False,
                "message": error_msg,
                "text": "",
                "accumulated_text": context["text"] if "context" in locals() and "text" in context else "",
                "is_final": True,
                "time_cost": stats.total_time() if stats else 0
            }
    
    def init_websocket_session(self, session_id: str, language: str = "auto", 
                              use_itn: bool = True) -> None:
        """
        初始化WebSocket会话
        
        参数:
            session_id: 会话ID
            language: 语言代码
            use_itn: 是否使用反向文本归一化
        """
        with self._ws_lock:
            self._accumulated_results[session_id] = {
                "buffer": [],  # 音频数据缓冲
                "text": "",    # 累积的文本结果
                "chunk_idx": 0,  # 块索引
                "language": language,  # 语言设置
                "use_itn": use_itn,    # 是否使用反向文本归一化
                "language_tag": None,  # 检测到的语言标签
                "emotion_tag": None,   # 情感标签
                "event_tag": None,     # 事件标签
                "temp_files": []       # 临时文件列表
            }
            logger.info(f"WebSocket会话初始化，会话ID: {session_id}, 语言: {language}, ITN: {use_itn}")
    
    def close_websocket_session(self, session_id: str) -> None:
        """
        关闭WebSocket会话并清理资源
        
        参数:
            session_id: 会话ID
        """
        with self._ws_lock:
            if session_id in self._accumulated_results:
                # 清理临时文件
                for temp_file in self._accumulated_results[session_id].get("temp_files", []):
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"清理临时文件失败: {temp_file}, 错误: {str(e)}")
                
                # 删除会话数据
                del self._accumulated_results[session_id]
                logger.info(f"WebSocket会话关闭，会话ID: {session_id}")
    
    def process_websocket_audio(self, session_id: str, audio_chunk: bytes) -> Tuple[bool, Dict[str, Any]]:
        """
        处理WebSocket音频数据块
        
        参数:
            session_id: 会话ID
            audio_chunk: 音频数据块
            
        返回:
            Tuple[bool, Dict]: 是否有新结果和结果字典
        """
        import wave
        import tempfile
        from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
        from processor import extract_tags
        
        if not self.is_loaded():
            return True, {
                "success": False,
                "message": "模型未加载",
                "text": "",
                "is_final": False
            }
        
        with self._ws_lock:
            if session_id not in self._accumulated_results:
                logger.error(f"未找到会话: {session_id}")
                return True, {
                    "success": False,
                    "message": f"未找到会话: {session_id}",
                    "text": "",
                    "is_final": False
                }
            
            session = self._accumulated_results[session_id]
            
            # 检查是否是空音频块
            if not audio_chunk or len(audio_chunk) == 0:
                # 空音频块通常表示音频流结束
                return True, {
                    "success": True,
                    "message": "识别完成",
                    "text": session["text"],
                    "language": session["language_tag"],
                    "emotion": session["emotion_tag"],
                    "event": session["event_tag"],
                    "is_final": True
                }
            
            try:
                # 将音频块追加到缓冲区
                session["buffer"].append(audio_chunk)
                
                # 音频块累积到一定大小时进行处理
                # 这里我们假设每个音频块大约是100ms的数据
                # 累积约1秒的数据再处理
                if len(session["buffer"]) >= 10:  # 可以根据实际情况调整
                    session["chunk_idx"] += 1
                    
                    # 创建临时WAV文件
                    temp_file = os.path.join(
                        config.TEMP_DIR, 
                        f"ws_{session_id}_{session['chunk_idx']}.wav"
                    )
                    session["temp_files"].append(temp_file)
                    
                    # 将音频数据写入临时文件
                    # 注意：这里假设传入的是原始PCM数据，实际应用中需根据客户端发送的格式调整
                    # 这里我们创建一个单声道16KHz的WAV文件
                    with wave.open(temp_file, 'wb') as wf:
                        wf.setnchannels(1)  # 单声道
                        wf.setsampwidth(2)  # 16位
                        wf.setframerate(self._sample_rate)  # 16KHz
                        for chunk in session["buffer"]:
                            wf.writeframes(chunk)
                    
                    # 清空缓冲区
                    session["buffer"] = []
                    
                    # 进行识别
                    inference_start = time.time()
                    results = self._model([temp_file], 
                                          language=session["language"], 
                                          use_itn=session["use_itn"])
                    inference_time = time.time() - inference_start
                    
                    # 清理临时文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        session["temp_files"].remove(temp_file)
                    
                    # 处理识别结果
                    if results and len(results) > 0:
                        current_result = results[0]
                        
                        # 后处理文本
                        processed_text = rich_transcription_postprocess(current_result)
                        
                        # 提取标签
                        language_tag, emotion_tag, event_tag = extract_tags(current_result)
                        session["language_tag"] = language_tag
                        session["emotion_tag"] = emotion_tag
                        session["event_tag"] = event_tag
                        
                        # 更新累积文本
                        if not session["text"]:
                            session["text"] = processed_text
                        else:
                            session["text"] += " " + processed_text
                        
                        # 构建返回结果
                        result = {
                            "success": True,
                            "message": "部分识别结果",
                            "text": processed_text,
                            "accumulated_text": session["text"],
                            "language": language_tag,
                            "emotion": emotion_tag,
                            "event": event_tag,
                            "is_final": False,
                            "chunk_id": session["chunk_idx"],
                            "time_cost": inference_time
                        }
                        
                        return True, result
                
                # 如果缓冲区数据不足，先不处理
                return False, {}
                
            except Exception as e:
                logger.error(f"WebSocket音频处理异常: {str(e)}")
                return True, {
                    "success": False,
                    "message": f"处理异常: {str(e)}",
                    "text": session.get("text", ""),
                    "is_final": False
                }

# 创建全局模型管理器实例
model_manager = ModelManager() 