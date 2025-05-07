#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API模块 - 提供FastAPI接口服务
"""
import os
import time
import tempfile
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
from audio_processor import AudioProcessor, AudioCache

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

# 后台清理任务
def cleanup_temp_file(filepath: str):
    """清理临时文件"""
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
            logger.debug(f"已清理临时文件: {filepath}")
    except Exception as e:
        logger.error(f"清理临时文件失败: {filepath} - {e}")

# --- API端点 ---
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回API首页"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>SenseVoice API信息</title>
        </head>
        <body>
            <h1>SenseVoice语音识别API</h1>
            <p>这是SenseVoice语音识别服务</p>
            <a href='./docs'>API文档</a>
        </body>
    </html>
    """

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
        raise HTTPException(500, f"获取状态失败: {str(e)}")

@app.post("/transcribe")
@timer
async def transcribe(
    background_tasks: BackgroundTasks,
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
        
        # 计算音频哈希值
        audio_hash = AudioProcessor.compute_audio_hash(content)
        
        # 检查缓存
        cached_result = AudioCache.get_cached_result(audio_hash)
        if cached_result:
            # 缓存命中
            logger.info(f"缓存命中: {audio.filename}")
            
            # 记录缓存命中指标
            PerformanceMonitor.record_metric("cache_hit", 1)
            
            # 更新处理时间
            total_time = time.time() - start_time
            
            # 返回缓存结果
            return {
                "status": "success",
                "text": cached_result.get("text", ""),
                "cached": True,
                "processing_time": total_time,
                "device": cached_result.get("device", "")
            }
        
        # 缓存未命中，保存临时文件
        with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            # 添加清理任务
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            
            # 执行推理
            results = model([tmp_path], language=language, textnorm=textnorm)
            
            # 获取文本
            text = results[0] if results else ""
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 构建结果
            result = {
                "status": "success",
                "text": text,
                "cached": False,
                "processing_time": total_time,
                "device": device
            }
            
            # 缓存结果
            cache_result = {
                "text": text,
                "device": result["device"]
            }
            AudioCache.cache_result(audio_hash, cache_result)
            
            # 返回结果
            return result
        except Exception as e:
            logger.error(f"推理处理失败: {str(e)}")
            raise HTTPException(500, f"推理处理失败: {str(e)}")
        finally:
            # 确保清理临时文件
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        # 记录错误
        logger.error(f"转写失败: {str(e)}")
        
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
        # 处理音频文件
        temp_files = []
        audio_paths = []
        
        try:
            # 将文件保存到临时位置
            for i, file_content in enumerate(files):
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = temp_file.name
                temp_file.write(file_content)
                temp_file.close()
                
                # 记录临时文件和路径
                temp_files.append(temp_path)
                audio_paths.append(temp_path)
            
            # 处理键名
            if keys == "":
                key = ["wav_file_tmp_name"]
            else:
                key = keys.split(",")
            
            # 执行推理
            texts = model(audio_paths, language=lang, textnorm="noitn")
            
            # 构建结果
            result = []
            for i, text in enumerate(texts):
                audio_name = key[i] if i < len(key) else f"audio_{i}"
                # 处理文本
                raw_text = text
                clean_text = re.sub(regex, "", raw_text, 0, re.MULTILINE)
                formatted_text = rich_transcription_postprocess(raw_text)
                
                result.append({
                    "key": audio_name,
                    "text": formatted_text,
                    "raw_text": raw_text,
                    "clean_text": clean_text
                })
                
            return {"result": result}
        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"清理临时文件失败: {temp_path} - {e}")
    except Exception as e:
        # 记录错误
        logger.error(f"批量转写失败: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_config=None
    )