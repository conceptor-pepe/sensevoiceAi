import os
import torch
import asyncio
import logging
import torchaudio
import re
import time
import traceback
import tempfile
import threading
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from config import settings
from performance import measure_performance, performance_monitor
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_manager")

class ModelManager:
    """
    模型管理器类，负责模型的加载、管理和推理
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化模型管理器
        """
        if not self._initialized:
            self.device = settings.DEVICE
            self.model_dir = settings.MODEL_DIR
            self.executor = ThreadPoolExecutor(max_workers=4)  # 用于异步推理的线程池
            self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)  # 并发控制
            self.regex = r"<\|.*\|>"
            self._request_counter = 0  # 请求计数器
            self._error_counter = 0  # 错误计数器
            self.model = None
            self.model_type = None  # 'onnx' 或 'torch'
            self.postprocess = None
            self.is_loaded = False
            self._lock = threading.Lock()
            self._load_model()
            self._initialized = True
            logger.info(f"模型管理器初始化完成，使用设备：{self.device}")
    
    def _load_model(self):
        """
        加载模型
        """
        start_time = time.time()
        logger.info(f"正在加载模型，路径：{self.model_dir}，设备：{self.device}")
        try:
            # 设置环境变量
            os.environ["SENSEVOICE_DEVICE"] = self.device
            
            # 确保模型目录存在
            self._check_model_dir()
            
            # 尝试导入模型库
            try:
                # 首先尝试使用torch版本
                from funasr_torch import SenseVoiceSmall
                from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess
                
                self.model_type = "torch"
                logger.info("使用Torch版本的SenseVoice模型")
                
                # 实例化模型
                self.model = SenseVoiceSmall(
                    model_dir=self.model_dir,
                    batch_size=10,  # 可配置批量大小
                    device=self.device
                )
                
            except ImportError:
                # 如果torch版本不可用，使用onnx版本
                try:
                    from funasr_onnx import SenseVoiceSmall
                    from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
                    
                    self.model_type = "onnx"
                    logger.info("使用ONNX版本的SenseVoice模型")
                    
                    # 实例化模型
                    self.model = SenseVoiceSmall(
                        model_dir=self.model_dir,
                        batch_size=10,
                        quantize=True  # ONNX版本可以量化提高性能
                    )
                    
                except ImportError:
                    raise ImportError("无法导入SenseVoice模型，请安装funasr-torch或funasr-onnx")
            
            # 保存后处理函数
            self.postprocess = rich_transcription_postprocess
            
            end_time = time.time()
            load_time = end_time - start_time
            logger.info(f"模型加载成功，耗时：{load_time:.2f}秒，模型类型: {self.model_type}")
            self.is_loaded = True
        except Exception as e:
            end_time = time.time()
            load_time = end_time - start_time
            logger.error(f"模型加载失败: {str(e)}, 耗时：{load_time:.2f}秒")
            logger.error(f"错误详情: {traceback.format_exc()}")
            self.is_loaded = False
    
    def _check_model_dir(self):
        """
        检查模型目录，确保模型文件存在
        """
        # 检查是否使用相对路径
        if '/' not in self.model_dir:
            # 对于modelscope hub模型，检查~/.cache/modelscope目录
            home_dir = str(Path.home())
            cache_dir = os.path.join(home_dir, ".cache", "modelscope", "hub")
            model_path = os.path.join(cache_dir, self.model_dir)
            
            if not os.path.exists(model_path):
                logger.warning(f"模型目录不存在: {model_path}")
                logger.info("首次运行将自动下载模型，这可能需要一些时间...")
                logger.info("如果您无法访问互联网，请手动下载模型并放置在正确位置")
        else:
            # 检查绝对路径或相对路径
            if not os.path.exists(self.model_dir):
                logger.warning(f"模型目录不存在: {self.model_dir}")
                raise FileNotFoundError(f"模型目录未找到: {self.model_dir}")
    
    @measure_performance(lambda self, audio_files, keys, language: 
        {"files_count": len(audio_files), "language": language})
    async def process_audio(self, 
                          audio_files: List[bytes], 
                          keys: List[str], 
                          language: str = "auto") -> Dict[str, Any]:
        """
        处理音频文件批量并返回结果
        
        参数:
            audio_files: 音频文件字节列表
            keys: 每个音频的名称列表
            language: 音频内容的语言
            
        返回:
            包含识别结果的字典
        """
        # 检查模型是否已加载
        if not self.is_loaded or self.model is None:
            logger.error("模型未加载，无法处理音频")
            return {"error": "模型未加载"}
        
        # 检查输入有效性
        if not audio_files or len(audio_files) == 0:
            return {"error": "没有提供音频文件"}
        
        if len(audio_files) != len(keys):
            return {"error": "音频文件数量与键名数量不匹配"}
        
        # 获取请求ID
        request_id = self._get_next_request_id()
        total_size = sum(len(f) for f in audio_files)
        logger.info(f"请求[{request_id}]: 开始处理音频，数量={len(audio_files)}，总大小={total_size}字节，语言={language}")
        
        async with self.semaphore:
            logger.info(f"请求[{request_id}]: 获取到处理信号量，开始处理")
            try:
                result = await self._process_audio_internal(request_id, audio_files, keys, language)
                logger.info(f"请求[{request_id}]: 处理完成，结果大小={len(str(result))}字节")
                return result
            except Exception as e:
                self._error_counter += 1
                logger.error(f"请求[{request_id}]: 处理过程中出错: {str(e)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                raise
    
    async def _process_audio_internal(self,
                                     request_id: int,
                                     audio_files: List[bytes], 
                                     keys: List[str], 
                                     language: str) -> Dict[str, Any]:
        """
        内部处理音频的方法
        """
        try:
            logger.info(f"[{request_id}] 开始处理音频请求，语言: {language}, 文件数: {len(audio_files)}")
            start_time = time.time()
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存音频文件到临时目录
                temp_files = []
                for i, (audio_data, key) in enumerate(zip(audio_files, keys)):
                    # 使用安全的文件名
                    safe_key = "".join(c if c.isalnum() else "_" for c in key)
                    temp_file = os.path.join(temp_dir, f"{safe_key}_{i}.wav")
                    
                    with open(temp_file, 'wb') as f:
                        f.write(audio_data)
                    
                    temp_files.append(temp_file)
                
                io_end = time.time()
                logger.info(f"[{request_id}] 临时文件写入完成，耗时: {io_end - start_time:.4f}秒")
                
                # 调用模型进行推理
                logger.info(f"[{request_id}] 调用SenseVoice模型进行推理...")
                raw_results = self.model(temp_files, language=language, use_itn=True)
                
                # 后处理结果
                processed_results = [self.postprocess(r) for r in raw_results]
                
                inference_end = time.time()
                logger.info(f"[{request_id}] 模型推理完成，耗时: {inference_end - io_end:.4f}秒")
            
            # 构建结果
            results = []
            for i, (key, text) in enumerate(zip(keys, processed_results)):
                result = {
                    "key": key,
                    "text": text,
                    "raw_text": raw_results[i],  # 原始输出
                    "language": language  # 如果模型返回检测到的语言，可以使用它
                }
                results.append(result)
            
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] 推理和结果处理完成，总耗时: {total_time:.4f}秒")
            
            return {"result": results}
        
        except Exception as e:
            logger.error(f"[{request_id}] 处理音频时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"处理音频失败: {str(e)}"}
    
    @measure_performance(lambda self, audio_stream, language: {"language": language})
    async def process_audio_stream(self, 
                                 audio_stream, 
                                 language: str = "auto") -> Dict[str, Any]:
        """
        处理音频流并返回结果
        
        参数:
            audio_stream: 音频流
            language: 音频内容的语言
            
        返回:
            包含识别结果的字典
        """
        # 获取请求ID
        request_id = self._get_next_request_id()
        logger.info(f"流式请求[{request_id}]: 开始处理音频流，语言={language}")
        
        # 流式处理逻辑
        # TODO: 实现完整的流式处理
        chunk_size = settings.CHUNK_SIZE
        chunks = []
        total_bytes = 0
        chunk_count = 0
        
        # 从流中读取数据
        start_time = time.time()
        while True:
            try:
                chunk = await audio_stream.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                total_bytes += len(chunk)
                chunk_count += 1
                if chunk_count % 10 == 0:  # 每10个块记录一次
                    logger.info(f"流式请求[{request_id}]: 已读取 {chunk_count} 个块，总大小 {total_bytes} 字节")
            except Exception as e:
                logger.error(f"流式请求[{request_id}]: 读取音频流出错: {str(e)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                self._error_counter += 1
                raise
        
        read_time = time.time() - start_time
        logger.info(f"流式请求[{request_id}]: 音频流读取完成，共 {chunk_count} 个块，总大小 {total_bytes} 字节，耗时 {read_time:.4f} 秒")
        
        # 合并所有块
        audio_data = b''.join(chunks)
        
        # 使用常规处理函数处理合并后的数据
        result = await self.process_audio([audio_data], ["stream"], language)
        logger.info(f"流式请求[{request_id}]: 音频处理完成")
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        if not self.is_loaded:
            return {"status": "未加载", "error": "模型未成功加载"}
        
        info = {
            "status": "已加载",
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "device": self.device,
            "supported_languages": ["auto", "zh", "en", "yue", "ja", "ko"]
        }
        
        # 如果可能，添加更多模型信息
        try:
            # 尝试获取模型额外信息
            if hasattr(self.model, "get_model_info"):
                model_info = self.model.get_model_info()
                info.update(model_info)
        except:
            pass
        
        return info
    
    def _get_next_request_id(self) -> int:
        """
        获取下一个请求ID
        """
        with self._lock:
            self._request_counter += 1
            return self._request_counter
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        获取请求统计信息
        """
        return {
            "total_requests": self._request_counter,
            "error_requests": self._error_counter,
            "success_rate": f"{(1 - self._error_counter / max(1, self._request_counter)) * 100:.2f}%"
        }
    
    @property
    def is_ready(self) -> bool:
        """
        检查模型是否已准备好
        """
        return self.is_loaded and self.model is not None


# 创建全局模型管理器实例
model_manager = ModelManager() 