#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 接口模块
实现API路由和处理逻辑
"""

import os
import json
import uuid
from typing import Optional, List, Dict, Any, Union, Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
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

# 流式请求模型
class StreamRecognitionRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = "auto"
    use_itn: Optional[bool] = True
    chunk_size_sec: Optional[int] = 3

# WebSocket通信模型
class WebSocketConfig(BaseModel):
    language: Optional[str] = "auto"
    use_itn: Optional[bool] = True

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
    logger.info(f"request {request.method} {request.url.path} cost: {process_time:.4f}s")
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
                logger.error(f"[{stats.request_id}] request missing audio_base64 field")
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
            logger.info(f"[{stats.request_id}] received base64 audio data, size: {file_size/1024:.2f}KB")
        
        else:
            logger.error(f"[{stats.request_id}] no audio data provided")
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
        logger.error(f"[{stats.request_id}] process exception: {str(e)}")
        
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

@app.post("/recognize/stream")
async def recognize_audio_stream(
    audio_file: Optional[UploadFile] = File(None),
    request_data: Optional[str] = Form(None),
    language: Optional[str] = Form("auto"),
    use_itn: Optional[bool] = Form(True),
    chunk_size_sec: Optional[int] = Form(3)
):
    """
    流式语音识别API接口
    
    接受音频文件上传或base64编码的音频数据
    以流式方式返回识别结果、语言类型、情绪和事件类型
    """
    # 初始化时间统计
    stats = TimeStats(prefix="stream")
    
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
            logger.info(f"[{stats.request_id}] 接收到流式处理音频文件: {audio_file.filename}, 大小: {file_size/1024:.2f}KB")
            
        elif request_data is not None:
            # 尝试解析请求数据
            try:
                req_data = json.loads(request_data)
            except json.JSONDecodeError:
                req_data = {"audio_base64": request_data}
                
            # 验证请求数据
            if "audio_base64" not in req_data:
                logger.error(f"[{stats.request_id}] 缺少audio_base64字段")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": "缺少audio_base64字段", 
                        "time_cost": stats.total_time()
                    }
                )
                
            # 获取参数
            audio_base64 = req_data.get("audio_base64")
            language = req_data.get("language", language)
            use_itn = req_data.get("use_itn", use_itn)
            chunk_size_sec = req_data.get("chunk_size_sec", chunk_size_sec)
            
            # 解码base64数据
            audio_data = processor.decode_base64_audio(audio_base64, stats)
            if audio_data is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": "Base64解码失败", 
                        "time_cost": stats.total_time()
                    }
                )
                
            # 保存到临时文件
            audio_path = processor.save_temp_audio(audio_data)
            file_size = len(audio_data)
            logger.info(f"[{stats.request_id}] 接收到流式处理base64数据, 大小: {file_size/1024:.2f}KB")
            
        else:
            logger.error(f"[{stats.request_id}] 未提供音频数据")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False, 
                    "message": "请提供音频文件或base64编码的音频数据", 
                    "time_cost": stats.total_time()
                }
            )
            
        stats.record_step("数据准备")
        
        # 创建异步生成器
        async def generate_stream():
            """生成流式响应数据"""
            try:
                # 获取流式处理结果
                async for chunk in processor.process_stream_audio(
                    audio_path, language, use_itn, chunk_size_sec, stats
                ):
                    # 返回处理结果
                    yield chunk + "\n"
                    
                # 最后清理临时文件
                processor.cleanup_temp_files([audio_path])
                stats.record_step("清理和完成")
                
            except Exception as e:
                logger.error(f"[{stats.request_id}] 流式处理异常: {str(e)}")
                # 返回错误消息
                yield json.dumps({
                    "success": False,
                    "message": f"流式处理异常: {str(e)}",
                    "is_final": True,
                    "time_cost": stats.total_time()
                }) + "\n"
                
                # 尝试清理临时文件
                if audio_path and os.path.exists(audio_path):
                    processor.cleanup_temp_files([audio_path])
        
        # 返回流式响应
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson"  # 使用新行分隔的JSON格式
        )
        
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
                "time_cost": stats.total_time()
            }
        )

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    WebSocket流式语音识别接口
    
    通过WebSocket双向通信进行实时语音识别
    客户端可以持续发送音频数据流，服务端实时返回识别结果
    """
    session_id = str(uuid.uuid4())
    stats = TimeStats(prefix=f"ws_{session_id}")
    await websocket.accept()
    
    # 默认配置
    language = "auto"
    use_itn = True
    
    try:
        # 接收配置信息
        config_data = await websocket.receive_text()
        try:
            ws_config = json.loads(config_data)
            # 获取配置参数
            language = ws_config.get("language", language)
            use_itn = ws_config.get("use_itn", use_itn)
            logger.info(f"[{stats.request_id}] WebSocket配置: 语言={language}, ITN={use_itn}")
        except json.JSONDecodeError:
            logger.warning(f"[{stats.request_id}] 无法解析WebSocket配置, 使用默认配置")
        
        # 初始化WebSocket会话
        model_manager.init_websocket_session(session_id, language, use_itn)
        
        # 发送会话已准备好的消息
        await websocket.send_json({
            "type": "ready",
            "session_id": session_id,
            "message": "会话已准备好，请开始发送音频数据"
        })
        
        # 循环接收音频数据
        while True:
            # 接收音频二进制数据
            audio_data = await websocket.receive_bytes()
            
            # 处理音频数据
            has_result, result = model_manager.process_websocket_audio(session_id, audio_data)
            
            # 如果有新结果，发送给客户端
            if has_result:
                await websocket.send_json(result)
                
                # 如果是最终结果，结束会话
                if result.get("is_final", False):
                    break
    
    except WebSocketDisconnect:
        logger.info(f"[{stats.request_id}] WebSocket客户端断开连接")
    except Exception as e:
        logger.error(f"[{stats.request_id}] WebSocket错误: {str(e)}")
        try:
            await websocket.send_json({
                "success": False,
                "message": f"WebSocket错误: {str(e)}",
                "is_final": True
            })
        except:
            pass
    finally:
        # 清理资源
        model_manager.close_websocket_session(session_id)
        stats.log_stats(prefix="WebSocket会话结束，")

@app.websocket("/api/v1/ws/asr")
async def websocket_asr_v1(websocket: WebSocket):
    """
    与GitHub SenseVoice API兼容的WebSocket流式识别接口
    
    通过WebSocket通信进行实时语音识别，输出与其他API端点兼容的格式
    """
    session_id = str(uuid.uuid4())
    stats = TimeStats(prefix=f"ws_v1_{session_id}")
    await websocket.accept()
    
    # 默认配置
    language = "auto"
    use_itn = False  # v1接口默认关闭ITN
    
    try:
        # 接收配置信息
        config_data = await websocket.receive_text()
        try:
            ws_config = json.loads(config_data)
            # 获取配置参数
            language = ws_config.get("lang", language)
            use_itn = ws_config.get("use_itn", use_itn)
            key = ws_config.get("key", f"ws_{session_id}")
            logger.info(f"[{stats.request_id}] WebSocket V1配置: 语言={language}, ITN={use_itn}, key={key}")
        except json.JSONDecodeError:
            key = f"ws_{session_id}"
            logger.warning(f"[{stats.request_id}] 无法解析WebSocket配置, 使用默认配置")
        
        # 初始化WebSocket会话
        model_manager.init_websocket_session(session_id, language, use_itn)
        
        # 发送会话已准备好的消息
        await websocket.send_json({
            "status": "ready",
            "key": key,
            "message": "会话已准备好，请开始发送音频数据"
        })
        
        # 循环接收音频数据
        while True:
            # 接收音频二进制数据
            audio_data = await websocket.receive_bytes()
            
            # 处理音频数据
            has_result, result = model_manager.process_websocket_audio(session_id, audio_data)
            
            # 如果有新结果，转换为v1格式并发送给客户端
            if has_result:
                # 构造v1格式响应
                v1_result = {
                    "result": [{
                        "key": key,
                        "raw_text": result.get("text", ""),
                        "clean_text": result.get("text", ""),
                        "text": result.get("text", ""),
                        "language": result.get("language"),
                        "emotion": result.get("emotion"),
                        "event": result.get("event"),
                        "is_final": result.get("is_final", False),
                        "accumulated_text": result.get("accumulated_text", "")
                    }],
                    "is_final": result.get("is_final", False),
                    "time_cost": result.get("time_cost", 0)
                }
                
                if "detail_time" in result:
                    v1_result["detail_time"] = result["detail_time"]
                
                await websocket.send_json(v1_result)
                
                # 如果是最终结果，结束会话
                if result.get("is_final", False):
                    break
    
    except WebSocketDisconnect:
        logger.info(f"[{stats.request_id}] WebSocket V1客户端断开连接")
    except Exception as e:
        logger.error(f"[{stats.request_id}] WebSocket V1错误: {str(e)}")
        try:
            await websocket.send_json({
                "result": [],
                "message": f"WebSocket错误: {str(e)}",
                "is_final": True
            })
        except:
            pass
    finally:
        # 清理资源
        model_manager.close_websocket_session(session_id)
        stats.log_stats(prefix="WebSocket V1会话结束，")

@app.post("/api/v1/asr/stream")
async def turn_audio_to_text_stream(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")], 
    keys: Annotated[str, Form(description="name of each audio joined with comma")], 
    lang: Annotated[str, Form(description="language of audio content")] = "auto",
    use_itn: Annotated[bool, Form(description="whether to use inverse text normalization")] = False,
    chunk_size_sec: Annotated[int, Form(description="chunk size in seconds for streaming")] = 3
):
    """
    与GitHub SenseVoice API兼容的流式语音识别接口
    
    接受音频文件上传，以流式方式返回识别结果
    目前只支持单个文件的处理，如果有多个文件，只处理第一个
    """
    # 初始化时间统计
    stats = TimeStats(prefix="stream_batch")
    
    # 检查模型状态
    if not model_manager.is_loaded():
        logger.error(f"[{stats.request_id}] model not loaded")
        return JSONResponse(
            status_code=500,
            content={
                "result": [], 
                "message": "模型未加载成功", 
                "time_cost": stats.total_time()
            }
        )
    
    try:
        # 验证是否有文件上传
        if not files or len(files) == 0:
            logger.error(f"[{stats.request_id}] no files uploaded")
            return JSONResponse(
                status_code=400,
                content={
                    "result": [], 
                    "message": "未上传文件", 
                    "time_cost": stats.total_time()
                }
            )
        
        # 获取第一个文件
        file = files[0]
        content = await file.read()
        file_size = len(content)
        
        # 保存到临时文件
        audio_path = processor.save_temp_audio(content, file.filename)
        stats.record_step("文件上传")
        logger.info(f"[{stats.request_id}] 接收到流式处理音频文件: {file.filename}, 大小: {file_size/1024:.2f}KB")
        
        # 获取键名
        key = keys.split(",")[0] if keys else f"file_0"
        
        # 创建异步生成器
        async def generate_stream():
            """生成流式响应数据"""
            try:
                # 处理上下文
                context = {
                    "key": key,
                    "raw_results": [],
                    "last_result": None
                }
                
                # 获取流式处理结果
                async for chunk_json in processor.process_stream_audio(
                    audio_path, lang, use_itn, chunk_size_sec, stats
                ):
                    # 解析JSON结果
                    chunk = json.loads(chunk_json)
                    
                    # 是否为最终结果
                    is_final = chunk.get("is_final", False)
                    
                    # 构建结果结构
                    result = {
                        "key": key,
                        "raw_text": chunk.get("text", ""),
                        "clean_text": chunk.get("text", ""),  # 简化处理
                        "text": chunk.get("text", ""),
                        "language": chunk.get("language"),
                        "emotion": chunk.get("emotion"),
                        "event": chunk.get("event"),
                        "is_final": is_final,
                        "accumulated_text": chunk.get("accumulated_text", "")
                    }
                    
                    # 包装为API响应格式
                    api_response = {
                        "result": [result],
                        "is_final": is_final,
                        "time_cost": chunk.get("time_cost", 0)
                    }
                    
                    # 最终结果添加详细时间统计
                    if is_final and "detail_time" in chunk:
                        api_response["detail_time"] = chunk["detail_time"]
                    
                    # 返回JSON格式的结果
                    yield json.dumps(api_response) + "\n"
                    
                    # 记录上下文
                    if is_final:
                        context["last_result"] = result
                
                # 清理临时文件
                processor.cleanup_temp_files([audio_path])
                stats.record_step("清理和完成")
                
            except Exception as e:
                logger.error(f"[{stats.request_id}] 流式处理异常: {str(e)}")
                # 返回错误消息
                yield json.dumps({
                    "result": [],
                    "message": f"流式处理异常: {str(e)}",
                    "is_final": True,
                    "time_cost": stats.total_time()
                }) + "\n"
                
                # 尝试清理临时文件
                if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                    processor.cleanup_temp_files([audio_path])
        
        # 返回流式响应
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson"  # 使用新行分隔的JSON格式
        )
                
    except Exception as e:
        logger.error(f"[{stats.request_id}] 处理异常: {str(e)}")
        
        # 尝试清理临时文件
        if 'audio_path' in locals() and audio_path:
            processor.cleanup_temp_files([audio_path])
            
        return JSONResponse(
            status_code=500,
            content={
                "result": [], 
                "message": f"处理失败: {str(e)}", 
                "time_cost": stats.total_time()
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
        logger.error(f"[{stats.request_id}] model not loaded")
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
        
        logger.info(f"[{stats.request_id}] received batch processing request, file count: {len(files)}")
        
        # 处理上传的文件
        for file in files:
            content = await file.read()
            file_sizes.append(len(content))
            audio_path = processor.save_temp_audio(content, file.filename)
            audio_paths.append(audio_path)
        
        stats.record_step("文件上传")
        logger.info(f"[{stats.request_id}] all files uploaded, total size: {sum(file_sizes)/1024:.2f}KB")
        
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
        logger.error(f"[{stats.request_id}] process exception: {str(e)}")
        
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
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
