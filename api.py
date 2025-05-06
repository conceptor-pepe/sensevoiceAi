#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 实现
支持标准API、流式识别和WebSocket实时交互
"""

import os
import json
import base64
import uuid
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from stats import TimeStats
from logger import logger
from processor import (save_temp_audio, cleanup_temp_files, decode_base64_audio,
                      process_single_audio, process_multiple_audio, process_stream_audio)
from model_manager import model_manager

# ================== 初始化FastAPI ==================
app = FastAPI(
    title="SenseVoice API",
    description="高性能语音识别API，支持流式识别和WebSocket实时交互",
    version="1.0.0"
)

# 添加CORS中间件支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# ================== 请求模型 ==================
class RecognizeRequest(BaseModel):
    """识别请求模型"""
    audio_base64: str = Field(..., description="Base64编码的音频数据")
    language: str = Field("auto", description="语言代码，auto自动检测")
    use_itn: bool = Field(True, description="是否使用反向文本归一化")
    
class StreamRequest(BaseModel):
    """流式识别请求模型"""
    audio_base64: str = Field(..., description="Base64编码的音频数据")
    language: str = Field("auto", description="语言代码，auto自动检测")
    use_itn: bool = Field(True, description="是否使用反向文本归一化")
    chunk_size_sec: int = Field(3, description="分块大小（秒）")
    
# ================== 辅助函数 ==================
def validate_language(language: str) -> str:
    """验证语言代码"""
    if language not in config.SUPPORTED_LANGUAGES:
        logger.warning(f"不支持的语言: {language}，将使用auto替代")
        return "auto"
    return language

# ================== 标准API接口 ==================
@app.get("/")
async def root():
    """API根路径，返回API信息"""
    return {
        "name": "SenseVoice API",
        "version": "1.0",
        "status": "running",
        "model": config.MODEL_DIR
    }

@app.post("/recognize")
async def recognize_audio(
    background_tasks: BackgroundTasks,
    audio_file: Optional[UploadFile] = File(None),
    language: str = Form("auto"),
    use_itn: bool = Form(True),
    request_data: Optional[str] = Form(None)
):
    """
    标准语音识别接口
    支持文件上传或Base64编码音频
    """
    # 创建统计对象
    stats = TimeStats("recognize")
    
    try:
        # 记录请求信息
        logger.info(f"[{stats.request_id}] 收到识别请求，语言：{language}, ITN：{use_itn}")
        
        # 检查模型是否已加载
        if not model_manager.is_loaded():
            model_manager.load_model()
        
        # 验证语言设置
        language = validate_language(language)
        
        # 音频文件路径列表
        temp_files = []
        
        # 处理音频文件上传
        if audio_file:
            stats.record_step("接收音频文件")
            content = await audio_file.read()
            
            if not content:
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "message": "上传的音频文件为空"
                })
                
            # 保存音频到临时文件
            audio_path = save_temp_audio(content, audio_file.filename)
            temp_files.append(audio_path)
            stats.record_step("保存音频文件")
            
        # 处理JSON请求数据（Base64编码）
        elif request_data:
            stats.record_step("接收JSON数据")
            
            try:
                data = json.loads(request_data)
                
                # 提取参数
                if "language" in data:
                    language = validate_language(data["language"])
                if "use_itn" in data:
                    use_itn = data["use_itn"]
                
                # 处理Base64编码的音频数据
                if "audio_base64" in data:
                    # 解码Base64
                    audio_data = decode_base64_audio(data["audio_base64"], stats)
                    if not audio_data:
                        return JSONResponse(status_code=400, content={
                            "success": False,
                            "message": "Base64解码失败"
                        })
                    
                    # 保存音频到临时文件
                    audio_path = save_temp_audio(audio_data)
                    temp_files.append(audio_path)
                    stats.record_step("保存音频文件")
                else:
                    return JSONResponse(status_code=400, content={
                        "success": False,
                        "message": "请求数据中未找到音频数据"
                    })
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "message": "无效的JSON格式"
                })
        else:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "请提供音频文件或Base64编码的音频数据"
            })
            
        # 调用处理器进行识别
        result = process_single_audio(audio_path, language, use_itn, stats)
        
        # 安排清理临时文件
        background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[{stats.request_id}] 识别失败: {str(e)}")
        
        # 清理临时文件
        if 'temp_files' in locals():
            background_tasks.add_task(cleanup_temp_files, temp_files)
            
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"处理失败: {str(e)}",
            "time_cost": stats.total_time()
        })

# ================== 兼容GitHub SenseVoice API接口 ==================
@app.post("/api/v1/asr")
async def api_v1_asr(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    keys: Optional[str] = Form("audio"),
    lang: str = Form("auto"),
    use_itn: bool = Form(True)
):
    """
    兼容GitHub SenseVoice API的接口
    支持批量处理多个音频文件
    """
    # 创建统计对象
    stats = TimeStats("api_v1")
    
    try:
        # 记录请求信息
        logger.info(f"[{stats.request_id}] 收到兼容API请求，文件数：{len(files)}, 语言：{lang}, ITN：{use_itn}")
        
        # 检查模型是否已加载
        if not model_manager.is_loaded():
            model_manager.load_model()
        
        # 验证语言设置
        language = validate_language(lang)
        
        # 解析keys
        key_list = keys.split(",") if keys else ["audio" + str(i) for i in range(len(files))]
        
        # 处理音频文件
        temp_files = []
        audio_paths = []
        
        for i, file in enumerate(files):
            content = await file.read()
            
            if not content:
                continue
                
            # 保存音频到临时文件
            audio_path = save_temp_audio(content, file.filename)
            audio_paths.append(audio_path)
            temp_files.append(audio_path)
        
        if not audio_paths:
            return JSONResponse(status_code=400, content={
                "result": [],
                "message": "未收到有效的音频文件"
            })
            
        stats.record_step("保存音频文件")
        
        # 调用处理器进行识别
        result = process_multiple_audio(audio_paths, key_list, language, use_itn, stats)
        
        # 安排清理临时文件
        background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[{stats.request_id}] 识别失败: {str(e)}")
        
        # 清理临时文件
        if 'temp_files' in locals():
            background_tasks.add_task(cleanup_temp_files, temp_files)
            
        return JSONResponse(status_code=500, content={
            "result": [],
            "message": f"处理失败: {str(e)}",
            "time_cost": stats.total_time()
        })

# ================== HTTP流式识别接口 ==================
@app.post("/recognize/stream")
async def recognize_stream(
    background_tasks: BackgroundTasks,
    audio_file: Optional[UploadFile] = File(None),
    language: str = Form("auto"),
    use_itn: bool = Form(True),
    chunk_size_sec: int = Form(3),
    request_data: Optional[str] = Form(None)
):
    """
    HTTP流式识别接口
    适用于离线音频文件的流式处理
    """
    # 创建统计对象
    stats = TimeStats("stream")
    
    # 音频文件路径列表
    temp_files = []
    
    try:
        # 记录请求信息
        logger.info(f"[{stats.request_id}] 收到流式识别请求，语言：{language}, ITN：{use_itn}, 块大小：{chunk_size_sec}秒")
        
        # 检查模型是否已加载
        if not model_manager.is_loaded():
            model_manager.load_model()
        
        # 验证语言设置
        language = validate_language(language)
        
        # 处理音频文件上传
        if audio_file:
            stats.record_step("接收音频文件")
            content = await audio_file.read()
            
            if not content:
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "message": "上传的音频文件为空"
                })
                
            # 保存音频到临时文件
            audio_path = save_temp_audio(content, audio_file.filename)
            temp_files.append(audio_path)
            stats.record_step("保存音频文件")
            
        # 处理JSON请求数据（Base64编码）
        elif request_data:
            stats.record_step("接收JSON数据")
            
            try:
                data = json.loads(request_data)
                
                # 提取参数
                if "language" in data:
                    language = validate_language(data["language"])
                if "use_itn" in data:
                    use_itn = data["use_itn"]
                if "chunk_size_sec" in data:
                    chunk_size_sec = data["chunk_size_sec"]
                
                # 处理Base64编码的音频数据
                if "audio_base64" in data:
                    # 解码Base64
                    audio_data = decode_base64_audio(data["audio_base64"], stats)
                    if not audio_data:
                        return JSONResponse(status_code=400, content={
                            "success": False,
                            "message": "Base64解码失败"
                        })
                    
                    # 保存音频到临时文件
                    audio_path = save_temp_audio(audio_data)
                    temp_files.append(audio_path)
                    stats.record_step("保存音频文件")
                else:
                    return JSONResponse(status_code=400, content={
                        "success": False,
                        "message": "请求数据中未找到音频数据"
                    })
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "message": "无效的JSON格式"
                })
        else:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "请提供音频文件或Base64编码的音频数据"
            })
            
        # 创建流式响应
        async def stream_generator():
            async for result in process_stream_audio(audio_path, language, use_itn, chunk_size_sec, stats):
                yield result + "\n"
            
            # 处理完成后清理临时文件
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        # 返回流式响应
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"  # 新行分隔的JSON
        )
        
    except Exception as e:
        logger.error(f"[{stats.request_id}] 流式识别失败: {str(e)}")
        
        # 清理临时文件
        if 'temp_files' in locals():
            background_tasks.add_task(cleanup_temp_files, temp_files)
            
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"处理失败: {str(e)}",
            "time_cost": stats.total_time()
        })

# ================== 兼容GitHub SenseVoice API流式接口 ==================
@app.post("/api/v1/asr/stream")
async def api_v1_asr_stream(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    keys: Optional[str] = Form("audio"),
    lang: str = Form("auto"),
    use_itn: bool = Form(True),
    chunk_size_sec: int = Form(3)
):
    """
    兼容GitHub SenseVoice API的流式接口
    支持离线音频文件的流式处理
    """
    # 创建统计对象
    stats = TimeStats("api_v1_stream")
    
    # 音频文件路径列表
    temp_files = []
    
    try:
        # 记录请求信息
        logger.info(f"[{stats.request_id}] 收到兼容流式API请求，文件数：{len(files)}, 语言：{lang}, ITN：{use_itn}")
        
        # 检查模型是否已加载
        if not model_manager.is_loaded():
            model_manager.load_model()
        
        # 验证语言设置
        language = validate_language(lang)
        
        # 确保只处理第一个文件
        if not files:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "未提供音频文件"
            })
            
        audio_file = files[0]
        content = await audio_file.read()
        
        if not content:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "上传的音频文件为空"
            })
            
        # 保存音频到临时文件
        audio_path = save_temp_audio(content, audio_file.filename)
        temp_files.append(audio_path)
        stats.record_step("保存音频文件")
        
        # 创建流式响应
        async def stream_generator():
            async for result in process_stream_audio(audio_path, language, use_itn, chunk_size_sec, stats):
                yield result + "\n"
            
            # 处理完成后清理临时文件
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        # 返回流式响应
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"  # 新行分隔的JSON
        )
        
    except Exception as e:
        logger.error(f"[{stats.request_id}] 流式识别失败: {str(e)}")
        
        # 清理临时文件
        if 'temp_files' in locals():
            background_tasks.add_task(cleanup_temp_files, temp_files)
            
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"处理失败: {str(e)}",
            "time_cost": stats.total_time()
        })

# ================== WebSocket接口 ==================
@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    WebSocket语音识别接口
    支持实时音频流处理
    """
    # 创建唯一会话ID
    session_id = f"ws_{uuid.uuid4().hex}"
    
    # 记录会话开始
    logger.info(f"WebSocket连接已建立，会话ID: {session_id}")
    
    # 检查模型是否已加载
    if not model_manager.is_loaded():
        model_manager.load_model()
    
    try:
        # 接受连接
        await websocket.accept()
        
        # 等待配置信息
        config_data = await websocket.receive_json()
        
        # 提取配置
        language = validate_language(config_data.get("language", "auto"))
        use_itn = config_data.get("use_itn", True)
        
        # 初始化WebSocket会话
        model_manager.init_websocket_session(session_id, language, use_itn)
        
        # 发送确认消息
        await websocket.send_json({
            "status": "ready",
            "message": "连接已建立，可以开始发送音频数据"
        })
        
        # 处理音频数据
        while True:
            # 接收音频数据块
            audio_data = await websocket.receive_bytes()
            
            # 处理音频数据
            has_result, result = model_manager.process_websocket_audio(session_id, audio_data)
            
            # 发送结果
            if has_result:
                await websocket.send_json(result)
                
                # 如果是最终结果，结束会话
                if result.get("is_final", False):
                    break
                    
    except Exception as e:
        error_msg = f"WebSocket处理异常: {str(e)}"
        logger.error(f"[{session_id}] {error_msg}")
        
        try:
            # 尝试发送错误消息
            await websocket.send_json({
                "success": False,
                "message": error_msg,
                "is_final": True
            })
        except:
            pass
    finally:
        # 关闭WebSocket会话
        model_manager.close_websocket_session(session_id)
        logger.info(f"WebSocket连接已关闭，会话ID: {session_id}")

# ================== 兼容GitHub SenseVoice API的WebSocket接口 ==================
@app.websocket("/api/v1/ws/asr")
async def api_v1_ws_asr(websocket: WebSocket):
    """
    兼容GitHub SenseVoice API的WebSocket接口
    支持实时音频流处理
    """
    # 创建唯一会话ID
    session_id = f"api_v1_ws_{uuid.uuid4().hex}"
    
    # 记录会话开始
    logger.info(f"WebSocket连接已建立，会话ID: {session_id}")
    
    # 检查模型是否已加载
    if not model_manager.is_loaded():
        model_manager.load_model()
    
    try:
        # 接受连接
        await websocket.accept()
        
        # 等待配置信息
        config_data = await websocket.receive_json()
        
        # 提取配置
        language = validate_language(config_data.get("lang", "auto"))
        use_itn = config_data.get("use_itn", True)
        key = config_data.get("key", "audio")
        
        # 初始化WebSocket会话
        model_manager.init_websocket_session(session_id, language, use_itn)
        
        # 发送确认消息
        await websocket.send_json({
            "status": "ready"
        })
        
        # 处理音频数据
        while True:
            # 接收音频数据块
            audio_data = await websocket.receive_bytes()
            
            # 处理音频数据
            has_result, result = model_manager.process_websocket_audio(session_id, audio_data)
            
            # 发送结果（转换为兼容格式）
            if has_result:
                # 转换为兼容格式
                compat_result = {
                    "result": [{
                        "key": key,
                        "text": result.get("text", ""),
                        "language": result.get("language", language),
                        "emotion": result.get("emotion", "NEUTRAL")
                    }] if result.get("success", False) else [],
                    "is_final": result.get("is_final", False)
                }
                
                await websocket.send_json(compat_result)
                
                # 如果是最终结果，结束会话
                if result.get("is_final", False):
                    break
                    
    except Exception as e:
        error_msg = f"WebSocket处理异常: {str(e)}"
        logger.error(f"[{session_id}] {error_msg}")
        
        try:
            # 尝试发送错误消息
            await websocket.send_json({
                "result": [],
                "error": error_msg,
                "is_final": True
            })
        except:
            pass
    finally:
        # 关闭WebSocket会话
        model_manager.close_websocket_session(session_id)
        logger.info(f"WebSocket连接已关闭，会话ID: {session_id}")

# ================== 健康检查接口 ==================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    is_model_loaded = model_manager.is_loaded()
    
    return {
        "status": "healthy" if is_model_loaded else "degraded",
        "model_loaded": is_model_loaded,
        "model_dir": config.MODEL_DIR,
        "gpu_device": config.GPU_DEVICE
    }

# # ================== 启动函数 ==================
# if __name__ == "__main__":
#     import uvicorn
#     # 确保模型加载
#     if not model_manager.is_loaded():
#         model_manager.load_model()
#     # 启动服务
#     uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)