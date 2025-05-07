#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API模块 - 提供FastAPI接口服务
"""
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO
from enum import Enum

import torch
import torchaudio
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing_extensions import Annotated
from model import SenseVoiceSmall  # 直接导入模型类
try:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    # 如果无法导入，提供一个简单的后处理函数
    def rich_transcription_postprocess(text):
        return text

import config
from logger import logger, timer, OperationLogger, PerformanceMonitor
from audio_processor import AudioProcessor, audio_processor

# 创建FastAPI应用
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加Gzip压缩支持
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 正则表达式模式：匹配特殊标记，用于后处理
regex = r"<\|.*\|>"

# 语言选项枚举
class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

# 初始化模型（全局单例）
try:
    model_dir = config.MODEL_NAME
    device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    logger.info(f"正在加载模型：{model_dir}，设备：{device}")
    model, kwargs = SenseVoiceSmall.from_pretrained(
        model=model_dir, 
        device=device,
        batch_size=config.MODEL_BATCH_SIZE,
        quantize=config.MODEL_QUANTIZE,
        download_dir=config.MODEL_CACHE_DIR
    )
    model.eval()
    logger.info(f"模型加载完成")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.get("/status")
async def get_status():
    """获取服务状态"""
    try:
        # 收集性能指标
        metrics = PerformanceMonitor.get_metrics()
        
        # 获取GPU状态
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "available": gpu_available,
            "device_count": torch.cuda.device_count() if gpu_available else 0,
            "current_device": device,
            "device_name": torch.cuda.get_device_name(0) if gpu_available else "N/A"
        }
        
        # 返回状态信息
        return {
            "status": "running",
            "version": config.API_VERSION,
            "gpu_status": gpu_info,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"获取状态失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"获取状态失败: {str(e)}")

@app.post("/transcribe")
@timer
async def transcribe(
    audio: UploadFile = File(..., description=f"音频文件({', '.join(config.SUPPORTED_AUDIO_FORMATS)})"),
    language: str = "auto",
    textnorm: str = "withitn"
):
    """
    转写音频文件
    
    - **audio**: 音频文件
    - **language**: 语言设置 (auto/zh/en/ja...)
    - **textnorm**: 文本规范化 (withitn/noitn)
    """
    start_time = time.time()
    
    try:
        # 验证文件类型
        if not audio.filename.lower().endswith(config.SUPPORTED_AUDIO_FORMATS):
            raise HTTPException(
                400, 
                f"仅支持以下格式: {', '.join(config.SUPPORTED_AUDIO_FORMATS)}"
            )
        
        # 读取音频内容
        content = await audio.read()
        
        # 直接处理音频数据，无需临时文件
        try:
            # 处理音频数据（避免写入临时文件）
            audio_tensor = audio_processor.process_audio_bytes(content)
            
            # 直接使用模型推理
            results = model.inference([audio_tensor], language=language, use_itn=(textnorm == "withitn"))
            
            # 获取文本
            text = results[0][0]["text"] if results and results[0] else ""
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 构建结果
            result = {
                "status": "success",
                "text": text,
                "processing_time": total_time,
                "device": device
            }
            
            # 记录操作日志
            OperationLogger.log_operation(
                operation="转写请求",
                status="success",
                details={
                    "filename": audio.filename,
                    "language": language,
                    "processing_time": total_time
                }
            )
            
            # 返回结果
            return result
        except Exception as e:
            logger.error(f"推理处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(500, f"推理处理失败: {str(e)}")
    except Exception as e:
        # 记录错误
        logger.error(f"转写失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 记录操作日志
        OperationLogger.log_operation(
            operation="转写请求",
            status="failure",
            details={"filename": audio.filename if audio else "未知文件"},
            error=e
        )
        
        # 返回错误响应
        raise HTTPException(500, str(e))

@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[bytes], File(description="wav或mp3音频文件，16KHz采样率")], 
    keys: Annotated[str, Form(description="用逗号分隔的音频名称")] = "", 
    lang: Annotated[Language, Form(description="音频内容的语言")] = "auto"
):
    """
    批量转写音频文件 (兼容接口)
    
    - **files**: 音频文件列表
    - **keys**: 文件名列表（逗号分隔）
    - **lang**: 语言设置
    """
    try:
        start_time = time.time()
        
        # 解析键名
        key_list = keys.split(",") if keys else []
        
        # 直接处理所有音频数据
        audio_tensors = []
        for file_data in files:
            audio_tensor = audio_processor.process_audio_bytes(file_data)
            audio_tensors.append(audio_tensor)
        
        # 批量推理
        results = model.inference(
            audio_tensors, 
            language=lang, 
            use_itn=True, 
            key=key_list
        )
        
        # 格式化结果
        response = []
        if results and results[0]:
            for item in results[0]:
                response.append({
                    "key": item.get("key", ""),
                    "value": item.get("text", "")
                })
        
        # 计算总处理时间
        total_time = time.time() - start_time
        PerformanceMonitor.record_metric("batch_processing_time", total_time)
        
        # 记录操作日志
        OperationLogger.log_operation(
            operation="批量转写",
            status="success",
            details={
                "file_count": len(files),
                "language": lang,
                "processing_time": total_time
            }
        )
        
        return response
    except Exception as e:
        # 记录错误
        logger.error(f"批量转写失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 记录操作日志
        OperationLogger.log_operation(
            operation="批量转写",
            status="failure",
            error=e
        )
        
        # 返回错误响应
        raise HTTPException(500, str(e))

# if __name__ == "__main__":
#     import uvicorn
#     logger.info(f"启动API服务 127.0.0.1:8000")
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000,
#         workers=1,
#         log_config=None
#     )