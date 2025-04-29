import os
import re
import uuid
import asyncio
import threading
import time
import queue  # 导入标准队列模块
from typing import List, Dict, Optional
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import Annotated
from io import BytesIO

# 配置类
class Config:
    MODEL_DIR = os.getenv("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
    DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    WORKERS = int(os.getenv("SENSEVOICE_WORKERS", 4))
    TIMEOUT = int(os.getenv("SENSEVOICE_TIMEOUT", 30))  # 任务超时时间(秒)
    MAX_QUEUE_SIZE = int(os.getenv("SENSEVOICE_MAX_QUEUE", 100))  # 最大队列长度

# 内存存储实现
class MemoryStorage:
    def __init__(self):
        self.results = defaultdict(dict)
        self.lock = threading.Lock()
        # 使用标准库的 queue.Queue 代替 asyncio.Queue
        self.task_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)

    # 修改为同步方法，不再使用async
    def add_task(self, data: dict) -> str:
        task_id = str(uuid.uuid4())
        self.task_queue.put({"task_id": task_id, **data})
        return task_id

    def set_result(self, task_id: str, result: dict):
        with self.lock:
            self.results[task_id] = result

    async def get_result(self, task_id: str, timeout: int = Config.TIMEOUT) -> Optional[dict]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if task_id in self.results:
                    result = self.results.pop(task_id)
                    return result
            await asyncio.sleep(0.1)
        return None

# 工作器实现
class Worker:
    def __init__(self, storage: MemoryStorage, worker_id: int):
        self.storage = storage
        self.worker_id = worker_id
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        # 不再需要为每个线程创建事件循环
        while self.running:
            try:
                # 使用阻塞方式获取任务
                task = self.storage.task_queue.get()
                task_id = task["task_id"]
                
                # 模拟处理（替换为实际模型推理）
                audios = self._preprocess_audio(task["audio_data"])
                result = {"text": "模拟结果", "task_id": task_id}
                
                self.storage.set_result(task_id, result)
                self.storage.task_queue.task_done()

            except Exception as e:
                print(f"Worker {self.worker_id} error: {str(e)}")

    def _preprocess_audio(self, audio_data_list: List[bytes]) -> List[torch.Tensor]:
        # 实际实现需替换为音频预处理逻辑
        return []

# FastAPI 应用
app = FastAPI()
storage = MemoryStorage()
workers = [Worker(storage, i) for i in range(Config.WORKERS)]

@app.post("/asr")
async def recognize_speech(
    background_tasks: BackgroundTasks,
    files: List[bytes] = File(...),
    language: str = Form("auto")
):
    # 由于add_task不再是异步方法，移除await
    task_id = storage.add_task({
        "audio_data": files,
        "language": language
    })
    
    background_tasks.add_task(wait_and_clean, task_id)
    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = await storage.get_result(task_id)
    if not result:
        raise HTTPException(404, "Result not found or expired")
    return result

async def wait_and_clean(task_id: str):
    await asyncio.sleep(Config.TIMEOUT)
    with storage.lock:
        if task_id in storage.results:
            storage.results.pop(task_id)