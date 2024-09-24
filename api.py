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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import torch

import config
from logger import logger, timer, OperationLogger, PerformanceMonitor
from audio_processor import AudioProcessor, AudioCache
from model_manager import ModelManager

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

# 获取模型管理器实例
def get_model_manager():
    """依赖注入: 获取模型管理器实例"""
    return ModelManager()

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
@app.get("/status")
async def get_status():
    """获取服务状态"""
    try:
        # 收集性能指标
        metrics = PerformanceMonitor.get_metrics()
        
        # 获取GPU状态
        gpu_info = ModelManager.verify_gpu()
        
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
    textnorm: str = "withitn",
    model_manager: ModelManager = Depends(get_model_manager)
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
            
            # 直接返回字典而不是JSONResponse
            return {
                "status": "success",
                "text": cached_result.get("text", ""),
                "cached": True,
                "processing_time": total_time,
                "device": cached_result.get("device", "")
            }
        
        # 缓存未命中，处理音频
        with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 添加清理任务
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            
            # 处理音频
            audio_data, sr = AudioProcessor.process_audio(tmp_path)
            
            # 保存处理后的音频
            processed_path = AudioProcessor.save_processed_audio(audio_data, sr)
            
            # 添加清理任务
            background_tasks.add_task(cleanup_temp_file, processed_path)
            
            # 直接同步调用（最简单的方法，用于调试）
            results = model_manager.transcribe([processed_path], language=language, textnorm=textnorm)
            
            # 获取转写结果
            text = results[0] if results else ""
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 构建结果
            result = {
                "status": "success",
                "text": text,
                "cached": False,
                "processing_time": total_time,
                "sample_rate": f"{sr}Hz",
                "device": f"GPU{config.GPU_DEVICE_ID}({torch.cuda.get_device_name(0)})"
            }
            
            # 缓存结果
            cache_result = {
                "text": text,
                "device": result["device"]
            }
            AudioCache.cache_result(audio_hash, cache_result)
            
            # 直接返回字典而不是JSONResponse
            return result
        finally:
            # 确保清理临时文件
            if os.path.exists(tmp_path):
                try:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_config=None
    )