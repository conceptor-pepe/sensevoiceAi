#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 接口模块
实现API路由和处理逻辑
"""

import os
import json
from typing import Optional, List, Dict, Any, Union, Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from logger import logger
from stats import TimeStats
from model_manager import model_manager
import processor

# 请求模型
class RecognitionRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = "auto"  # 可选参数，默认为auto
    use_itn: Optional[bool] = True     # 是否使用反向文本归一化

# 响应模型
class RecognitionResponse(BaseModel):
    success: bool
    message: str
    text: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[str] = None
    event: Optional[str] = None
    time_cost: float
    detail_time: Optional[Dict[str, float]] = None  # 添加详细时间统计字段

# 创建FastAPI应用
app = FastAPI(
    title="SenseVoice API",
    description="SenseVoice Small模型ONNX部署API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求耗时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    记录每个请求的处理时间，并在响应头中返回
    """
    start_time = TimeStats()
    response = await call_next(request)
    process_time = start_time.total_time()
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"请求 {request.method} {request.url.path} 处理耗时: {process_time:.4f}秒")
    return response

@app.on_event("startup")
async def startup_event():
    """
    服务启动时加载模型
    """
    config.print_config()
    model_manager.load_model()

@app.get("/")
async def root():
    """
    健康检查接口
    """
    return {"status": "ok", "message": "SenseVoice API服务运行正常"}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_audio(
    audio_file: Optional[UploadFile] = File(None),
    request_data: Optional[str] = Form(None),
    language: Optional[str] = Form("auto"),
    use_itn: Optional[bool] = Form(True)
):
    """
    语音识别API接口
    
    接受音频文件上传或base64编码的音频数据
    返回识别结果、语言类型、情绪和事件类型
    """
    # 初始化时间统计
    stats = TimeStats()
    
    # 检查模型状态
    if not model_manager.is_loaded():
        logger.error(f"[{stats.request_id}] 模型未加载成功")
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "message": "模型未加载成功", 
                "time_cost": stats.total_time()
            }
        )
    
    try:
        # 处理音频文件
        audio_path = None
        
        if audio_file is not None:
            # 从上传文件获取音频
            content = await audio_file.read()
            audio_path = processor.save_temp_audio(content, audio_file.filename)
            file_size = len(content)
            stats.record_step("文件上传")
            logger.info(f"[{stats.request_id}] 接收到音频文件: {audio_file.filename}, 大小: {file_size/1024:.2f}KB")
        
        elif request_data is not None:
            # 从请求数据获取base64编码的音频
            try:
                req_data = json.loads(request_data)
            except json.JSONDecodeError:
                req_data = {"audio_base64": request_data}
            
            if "audio_base64" not in req_data:
                logger.error(f"[{stats.request_id}] 请求缺少audio_base64字段")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": "缺少audio_base64字段", 
                        "time_cost": stats.total_time(),
                        "detail_time": stats.get_stats()
                    }
                )
            
            # 从请求数据中获取参数
            audio_base64 = req_data.get("audio_base64")
            language = req_data.get("language", language)
            use_itn = req_data.get("use_itn", use_itn)
            
            # 解码base64数据
            audio_data = processor.decode_base64_audio(audio_base64, stats)
            if audio_data is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": "Base64解码失败", 
                        "time_cost": stats.total_time(),
                        "detail_time": stats.get_stats()
                    }
                )
                
            # 保存到临时文件
            audio_path = processor.save_temp_audio(audio_data)
            file_size = len(audio_data)
            logger.info(f"[{stats.request_id}] 接收到Base64音频数据, 大小: {file_size/1024:.2f}KB")
        
        else:
            logger.error(f"[{stats.request_id}] 未提供音频数据")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False, 
                    "message": "请提供音频文件或base64编码的音频数据", 
                    "time_cost": stats.total_time(),
                    "detail_time": stats.get_stats()
                }
            )
        
        stats.record_step("数据准备")
        
        # 处理音频
        result = processor.process_single_audio(audio_path, language, use_itn, stats)
        
        # 清理临时文件
        processor.cleanup_temp_files([audio_path])
        stats.record_step("清理和完成")
        
        # 记录完成日志
        stats.log_stats(prefix="处理完成，")
        
        return result
            
    except Exception as e:
        logger.error(f"[{stats.request_id}] 处理异常: {str(e)}")
        
        # 尝试清理临时文件
        if 'audio_path' in locals() and audio_path:
            processor.cleanup_temp_files([audio_path])
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "message": f"处理失败: {str(e)}", 
                "time_cost": stats.total_time(),
                "detail_time": stats.get_stats()
            }
        )

@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")], 
    keys: Annotated[str, Form(description="name of each audio joined with comma")], 
    lang: Annotated[str, Form(description="language of audio content")] = "auto",
    use_itn: Annotated[bool, Form(description="whether to use inverse text normalization")] = False
):
    """
    与GitHub SenseVoice API兼容的语音识别接口
    
    接受多个音频文件上传，返回识别结果（包含原始文本、清洗后文本和处理后文本）
    """
    # 初始化时间统计
    stats = TimeStats(prefix="batch")
    
    # 检查模型状态
    if not model_manager.is_loaded():
        logger.error(f"[{stats.request_id}] 模型未加载成功")
        return JSONResponse(
            status_code=500,
            content={
                "result": [], 
                "message": "模型未加载成功", 
                "time_cost": stats.total_time(),
                "detail_time": stats.get_stats()
            }
        )
    
    try:
        # 处理输入数据
        audio_paths = []
        file_sizes = []
        
        logger.info(f"[{stats.request_id}] 接收到批量处理请求，文件数: {len(files)}")
        
        # 处理上传的文件
        for file in files:
            content = await file.read()
            file_sizes.append(len(content))
            audio_path = processor.save_temp_audio(content, file.filename)
            audio_paths.append(audio_path)
        
        stats.record_step("文件上传")
        logger.info(f"[{stats.request_id}] 所有文件上传完成，总大小: {sum(file_sizes)/1024:.2f}KB")
        
        # 处理键值
        if keys == "":
            key_list = [f"file_{i}" for i in range(len(files))]
        else:
            key_list = keys.split(",")
            # 如果键名不足，自动补充
            while len(key_list) < len(files):
                key_list.append(f"file_{len(key_list)}")
        
        # 批量处理音频
        result = processor.process_multiple_audio(audio_paths, key_list, lang, use_itn, stats)
        
        # 清理临时文件
        processor.cleanup_temp_files(audio_paths)
        stats.record_step("清理和完成")
        
        # 记录完成日志
        stats.log_stats(prefix="批量处理完成，")
        
        return result
            
    except Exception as e:
        logger.error(f"[{stats.request_id}] 处理异常: {str(e)}")
        
        # 尝试清理临时文件
        if 'audio_paths' in locals():
            processor.cleanup_temp_files(audio_paths)
            
        return JSONResponse(
            status_code=500,
            content={
                "result": [], 
                "message": f"处理失败: {str(e)}", 
                "time_cost": stats.total_time(),
                "detail_time": stats.get_stats()
            }
        )

if __name__ == "__main__":
    # 服务启动配置
    HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")
    PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))
    
    # 启动信息
    print(f"================================")
    print(f"SenseVoice API 服务启动")
    print(f"主机: {HOST}")
    print(f"端口: {PORT}")
    print(f"================================")
    
    # 启动服务
    uvicorn.run(app, host=HOST, port=PORT)
