import os
import uuid
import threading
import time
import asyncio
import queue
from typing import List, Optional, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import Annotated
from io import BytesIO
from starlette.concurrency import run_in_threadpool

# 配置类
class Config:
    MODEL_DIR = os.getenv("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
    DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    WORKERS = int(os.getenv("SENSEVOICE_WORKERS", 4))           # 工作线程数
    MAX_WORKERS = int(os.getenv("SENSEVOICE_MAX_WORKERS", 20))  # 最大线程池工作者数量
    MAX_QUEUE_SIZE = int(os.getenv("SENSEVOICE_MAX_QUEUE", 100))  # 最大请求队列长度
    REQUEST_TIMEOUT = int(os.getenv("SENSEVOICE_REQUEST_TIMEOUT", 30))  # 请求超时时间(秒)
    BATCH_SIZE = int(os.getenv("SENSEVOICE_BATCH_SIZE", 16))    # 批处理大小

# 请求限流器
class RateLimiter:
    def __init__(self, max_concurrent: int = Config.WORKERS):
        """
        初始化请求限流器
        
        参数:
            max_concurrent: 最大并发请求数
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def acquire(self):
        """获取执行许可"""
        return await self.semaphore.acquire()
        
    def release(self):
        """释放执行许可"""
        self.semaphore.release()

# 模型类
class SenseVoiceModel:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SenseVoiceModel, cls).__new__(cls)
                cls._instance._init_model()
            return cls._instance
    
    def _init_model(self):
        """初始化模型"""
        # 这里应该是实际的模型初始化代码
        self.device = torch.device(Config.DEVICE)
        print(f"正在加载模型 {Config.MODEL_DIR} 到设备 {Config.DEVICE}...")
        # TODO: 替换为实际的模型加载代码
        # self.model = ...
        # self.model.to(self.device)
        
        # 初始化处理队列和工作线程
        self.task_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.results = {}
        self.result_lock = threading.Lock()
        self.worker_threads = []
        
        # 启动工作线程
        for i in range(Config.WORKERS):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
            
        self.initialized = True
        print(f"模型加载完成，启动了 {Config.WORKERS} 个工作线程")
    
    def _worker_loop(self):
        """工作线程循环处理队列中的任务"""
        while True:
            try:
                # 从队列获取任务
                task_id, audio_tensor, language = self.task_queue.get()
                
                # 执行转录
                result = self._process_transcription(audio_tensor, language)
                
                # 保存结果
                with self.result_lock:
                    self.results[task_id] = result
                
                # 标记任务完成
                self.task_queue.task_done()
            
            except Exception as e:
                print(f"工作线程错误: {str(e)}")
    
    def _process_transcription(self, audio_tensor: torch.Tensor, language: str) -> str:
        """
        处理实际的转录任务
        
        参数:
            audio_tensor: 音频张量
            language: 语言选项
            
        返回:
            转录文本
        """
        # TODO: 实际的音频转文本推理代码
        # 目前返回模拟结果
        time.sleep(0.5)  # 模拟推理耗时
        return f"模拟转录结果 - 语言: {language}"
    
    def transcribe_async(self, audio_tensor: torch.Tensor, language: str = "auto") -> str:
        """
        异步提交转录任务
        
        参数:
            audio_tensor: 音频张量
            language: 语言选项，默认为自动检测
            
        返回:
            任务ID
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 添加到处理队列
        self.task_queue.put((task_id, audio_tensor, language))
        
        return task_id
    
    def get_result(self, task_id: str, timeout: int = Config.REQUEST_TIMEOUT) -> Optional[str]:
        """
        获取转录结果
        
        参数:
            task_id: 任务ID
            timeout: 超时时间(秒)
            
        返回:
            转录结果，如果超时则返回None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查结果是否可用
            with self.result_lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            
            # 短暂休眠避免忙等
            time.sleep(0.1)
        
        return None
    
    def transcribe(self, audio_tensor: torch.Tensor, language: str = "auto") -> str:
        """
        同步转录（提交任务并等待结果）
        
        参数:
            audio_tensor: 音频张量
            language: 语言选项，默认为自动检测
            
        返回:
            转录文本
        """
        # 提交异步任务
        task_id = self.transcribe_async(audio_tensor, language)
        
        # 等待结果
        result = self.get_result(task_id)
        if result is None:
            raise TimeoutError("转录任务超时")
        
        return result

# 音频处理工具
class AudioProcessor:
    @staticmethod
    def load_audio(audio_data: bytes) -> torch.Tensor:
        """
        从字节数据加载音频
        
        参数:
            audio_data: 音频文件的字节数据
        
        返回:
            音频张量
        """
        # 从字节数据加载音频
        buffer = BytesIO(audio_data)
        waveform, sample_rate = torchaudio.load(buffer)
        
        # TODO: 添加其他必要的预处理步骤
        # 例如: 重采样、正规化等
        
        return waveform

# 创建线程池和限流器
thread_pool = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
rate_limiter = RateLimiter(max_concurrent=Config.WORKERS)

# FastAPI 应用
app = FastAPI(
    title="SenseVoice API",
    description="语音识别API服务 - 高并发版本",
    version="1.0.0"
)

# 响应模型
class RecognitionResponse(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str

# 中间件：请求计数和限流
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """请求处理中间件，添加处理时间头部"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# API 路由
@app.post("/asr", response_model=RecognitionResponse)
async def recognize_speech(
    files: List[bytes] = File(..., description="音频文件"),
    language: str = Form("auto", description="语言选项，auto为自动检测")
):
    """
    语音识别接口 - 同步版本
    
    接收音频文件并直接返回识别结果
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供音频文件")
    
    try:
        # 获取限流许可
        await rate_limiter.acquire()
        
        try:
            # 处理第一个音频文件
            audio_data = files[0]
            
            # 在线程池中执行CPU密集型操作
            audio_tensor = await run_in_threadpool(
                AudioProcessor.load_audio, 
                audio_data
            )
            
            # 获取模型单例并在线程池中执行推理
            model = SenseVoiceModel()
            text = await run_in_threadpool(
                model.transcribe, 
                audio_tensor, 
                language
            )
            
            # 返回结果
            return {"text": text}
        
        finally:
            # 释放限流许可
            rate_limiter.release()
    
    except TimeoutError as e:
        raise HTTPException(
            status_code=408, 
            detail=f"请求处理超时: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"处理请求时发生错误: {str(e)}"
        )

@app.post("/asr/async", response_model=TaskResponse)
async def recognize_speech_async(
    files: List[bytes] = File(..., description="音频文件"),
    language: str = Form("auto", description="语言选项，auto为自动检测")
):
    """
    语音识别接口 - 异步版本
    
    接收音频文件并返回任务ID，客户端可通过任务ID查询结果
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供音频文件")
    
    try:
        # 处理第一个音频文件
        audio_data = files[0]
        
        # 在线程池中执行CPU密集型操作
        audio_tensor = await run_in_threadpool(
            AudioProcessor.load_audio, 
            audio_data
        )
        
        # 获取模型单例并提交异步任务
        model = SenseVoiceModel()
        task_id = await run_in_threadpool(
            model.transcribe_async, 
            audio_tensor, 
            language
        )
        
        # 返回任务ID
        return {"task_id": task_id}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"处理请求时发生错误: {str(e)}"
        )

@app.get("/result/{task_id}", response_model=RecognitionResponse)
async def get_result(task_id: str):
    """
    获取异步任务结果
    
    参数:
        task_id: 任务ID
    """
    try:
        # 获取模型单例并获取结果
        model = SenseVoiceModel()
        result = await run_in_threadpool(
            model.get_result, 
            task_id
        )
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail="任务结果不存在或已过期"
            )
        
        # 返回结果
        return {"text": result}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"获取结果时发生错误: {str(e)}"
        )

@app.get("/")
def health_check():
    """健康检查接口"""
    # 获取模型实例状态
    model = SenseVoiceModel()
    queue_size = model.task_queue.qsize()
    worker_count = len(model.worker_threads)
    
    return {
        "status": "ok", 
        "message": "SenseVoice API 服务运行正常",
        "stats": {
            "queue_size": queue_size,
            "workers": worker_count,
            "max_queue": Config.MAX_QUEUE_SIZE,
            "device": Config.DEVICE
        }
    }