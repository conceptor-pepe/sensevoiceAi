# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile, Header, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import logging
import asyncio
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from model_manager import model_manager
from cache import cache_manager
from config import settings
from performance import performance_monitor, measure_performance
from monitoring import system_monitor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

# 定义语言枚举
class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

# 定义响应格式枚举
class ResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"

# 定义请求体模型
class TranscriptionRequest(BaseModel):
    model: str = Field(default="sense-voice-small", description="模型ID")
    language: Language = Field(default="auto", description="音频内容的语言")
    response_format: ResponseFormat = Field(default="json", description="响应格式")

# 定义流式响应模型
class StreamResponse(BaseModel):
    text: str
    is_final: bool

# 定义性能查询参数
class PerformanceQueryParams(BaseModel):
    function_name: Optional[str] = None
    time_window: Optional[int] = None

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="高性能语音识别API服务",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API密钥验证依赖
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    验证API密钥
    """
    if settings.API_KEY_ENABLED:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API密钥"
            )
    return x_api_key

# 健康检查端点
@app.get("/api/v1/health")
async def health_check():
    """
    健康检查API
    """
    health_status = {
        "status": "ok" if model_manager.is_ready else "unavailable",
        "version": settings.APP_VERSION,
    }
    return health_status

# 性能监控端点
@app.get("/api/v1/performance")
async def get_performance_stats(
    function_name: Optional[str] = None,
    time_window: Optional[int] = None
):
    """
    获取性能统计数据
    
    参数:
        function_name: 可选，指定函数名称
        time_window: 可选，时间窗口（秒）
    """
    if function_name:
        stats = performance_monitor.get_statistics(
            function_name=function_name,
            time_window=time_window
        )
        return {"function": function_name, "stats": stats}
    else:
        stats = performance_monitor.get_function_statistics()
        return {"functions": stats}

# 性能统计摘要端点
@app.get("/api/v1/performance/summary")
async def get_performance_summary():
    """
    获取性能统计摘要
    """
    # 打印摘要到日志
    performance_monitor.print_summary()
    
    # 1小时、24小时和全部时间的统计
    hour_stats = performance_monitor.get_function_statistics()
    day_stats = performance_monitor.get_function_statistics()
    all_stats = performance_monitor.get_function_statistics()
    
    return {
        "last_hour": hour_stats,
        "last_day": day_stats,
        "all_time": all_stats
    }

# 清除旧性能指标端点
@app.post("/api/v1/performance/clear")
async def clear_old_performance_metrics(max_age_hours: int = 24):
    """
    清除旧的性能指标
    
    参数:
        max_age_hours: 要保留的最大小时数
    """
    count = performance_monitor.clear_old_metrics(max_age_hours)
    return {"message": f"已清除 {count} 条旧性能指标"}

# 模型信息端点
@app.get("/api/v1/models")
async def get_models():
    """
    获取可用模型信息
    """
    model_info = model_manager.get_model_info()
    return {"models": [model_info]}

# 基础HTML页面
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    根路径返回简单HTML页面
    """
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>SenseVoice API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                a {
                    display: inline-block;
                    margin: 10px 0;
                    padding: 10px 15px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }
                a:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>SenseVoice API 服务</h1>
            <p>高性能语音识别API服务</p>
            <a href='./docs'>API文档</a>
        </body>
    </html>
    """

# 音频文件转录端点（类似OpenAI API）
@app.post("/api/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
@measure_performance(lambda file, model, language, response_format: 
    {"model": model, "language": language, "filename": getattr(file, "filename", "unknown")})
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件"),
    model: str = Form("sense-voice-small", description="要使用的模型ID"),
    language: Language = Form("auto", description="音频内容的语言"),
    response_format: ResponseFormat = Form("json", description="响应格式")
):
    """
    将音频文件转录为文本
    """
    # 记录请求信息
    logger.info(f"处理音频转录请求: 文件={file.filename}, 大小={file.size if hasattr(file, 'size') else '未知'}, 语言={language}, 格式={response_format}")
    
    # 检查文件是否为有效的音频格式
    valid_formats = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/webm"]
    file_content_type = file.content_type or "application/octet-stream"
    
    if not any(format in file_content_type for format in valid_formats):
        logger.warning(f"不支持的文件格式: {file_content_type}")
        # 尝试根据文件扩展名判断
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in [".wav", ".mp3", ".ogg", ".webm"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="不支持的文件格式。请上传WAV、MP3、OGG或WEBM格式的音频文件。"
            )
    
    # 读取文件内容
    content = await file.read()
    file_size = len(content)
    logger.info(f"读取文件完成: {file.filename}, 大小={file_size} 字节")
    
    # 检查缓存
    cached_result = await cache_manager.get(content, language)
    if cached_result:
        logger.info(f"缓存命中: {file.filename}")
        # 根据响应格式处理结果
        return format_response(cached_result, response_format)
    
    # 处理音频
    logger.info(f"开始处理音频: {file.filename}")
    result = await model_manager.process_audio([content], [file.filename or "audio"], language)
    logger.info(f"音频处理完成: {file.filename}, 结果长度={len(result.get('result', []))}")
    
    # 缓存结果
    await cache_manager.set(content, language, result)
    
    # 根据响应格式处理结果
    return format_response(result, response_format)

# 音频流式转录端点
@app.post("/api/v1/audio/transcriptions/stream", dependencies=[Depends(verify_api_key)])
@measure_performance(lambda request, model, language: {"model": model, "language": language})
async def transcribe_audio_stream(
    request: Request,
    model: str = Form("sense-voice-small", description="要使用的模型ID"),
    language: Language = Form("auto", description="音频内容的语言")
):
    """
    流式处理音频并返回转录文本
    """
    logger.info(f"处理流式音频转录请求: 语言={language}")
    
    # 处理音频流
    result = await model_manager.process_audio_stream(request.stream(), language)
    logger.info(f"流式音频处理完成: 结果长度={len(result.get('result', []))}")
    
    # 返回结果
    return StreamingResponse(
        content=stream_generator(result),
        media_type="application/json"
    )

# 传统API端点（兼容旧版）
@app.post("/api/v1/asr")
@measure_performance(lambda files, keys, lang: {"files_count": len(files), "language": lang})
async def turn_audio_to_text(
    files: Annotated[List[bytes], File(description="wav或mp3格式的16KHz音频")], 
    keys: Annotated[str, Form(description="以逗号分隔的音频名称")], 
    lang: Annotated[Language, Form(description="音频内容的语言")] = "auto"
):
    """
    将音频文件转录为文本（兼容旧版API）
    """
    logger.info(f"处理传统API转录请求: 文件数={len(files)}, 语言={lang}")
    
    # 处理音频文件
    key_list = keys.split(",") if keys else []
    
    # 检查缓存（仅对单个文件检查缓存）
    if len(files) == 1:
        cached_result = await cache_manager.get(files[0], lang)
        if cached_result:
            logger.info("缓存命中")
            return cached_result
    
    # 处理音频
    logger.info(f"开始处理音频文件: 数量={len(files)}")
    result = await model_manager.process_audio(files, key_list, lang)
    logger.info(f"音频处理完成: 结果长度={len(result.get('result', []))}")
    
    # 缓存结果（仅对单个文件缓存）
    if len(files) == 1:
        await cache_manager.set(files[0], lang, result)
    
    return result

# 格式化响应的辅助函数
def format_response(result: Dict[str, Any], response_format: ResponseFormat):
    """
    根据指定的格式格式化响应
    
    参数:
        result: 转录结果
        response_format: 响应格式
        
    返回:
        格式化后的响应
    """
    if response_format == ResponseFormat.json:
        return result
    
    elif response_format == ResponseFormat.text:
        # 只返回纯文本内容
        if not result.get("result"):
            return ""
        texts = [item.get("text", "") for item in result["result"] if "text" in item]
        return "\n".join(texts)
    
    elif response_format == ResponseFormat.srt:
        # 返回SRT格式的字幕
        if not result.get("result"):
            return ""
        
        srt_content = ""
        for i, item in enumerate(result["result"]):
            if "text" not in item or not item["text"]:
                continue
                
            # 获取起始和结束时间（假设是以秒为单位）
            start_time = item.get("start_time", i * 5)
            end_time = item.get("end_time", start_time + 5)
            
            # 转换为SRT时间格式 (HH:MM:SS,mmm)
            start_formatted = format_time_srt(start_time)
            end_formatted = format_time_srt(end_time)
            
            srt_content += f"{i+1}\n{start_formatted} --> {end_formatted}\n{item['text']}\n\n"
            
        return srt_content
    
    elif response_format == ResponseFormat.vtt:
        # 返回WebVTT格式的字幕
        if not result.get("result"):
            return "WEBVTT\n\n"
        
        vtt_content = "WEBVTT\n\n"
        for i, item in enumerate(result["result"]):
            if "text" not in item or not item["text"]:
                continue
                
            # 获取起始和结束时间（假设是以秒为单位）
            start_time = item.get("start_time", i * 5)
            end_time = item.get("end_time", start_time + 5)
            
            # 转换为VTT时间格式 (HH:MM:SS.mmm)
            start_formatted = format_time_vtt(start_time)
            end_formatted = format_time_vtt(end_time)
            
            vtt_content += f"{i+1}\n{start_formatted} --> {end_formatted}\n{item['text']}\n\n"
            
        return vtt_content
    
    # 默认返回JSON
    return result

# 格式化SRT时间的辅助函数
def format_time_srt(seconds):
    """
    将秒数转换为SRT格式的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

# 格式化VTT时间的辅助函数
def format_time_vtt(seconds):
    """
    将秒数转换为WebVTT格式的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

# 流式响应生成器
async def stream_generator(result):
    """
    生成流式响应
    """
    # 如果没有结果，直接返回空对象
    if not result.get("result"):
        yield '{"text":"","is_final":true}'.encode('utf-8')
        return
    
    # 逐条输出结果
    for i, item in enumerate(result["result"]):
        is_final = i == len(result["result"]) - 1
        stream_resp = StreamResponse(text=item.get("text", ""), is_final=is_final)
        yield (stream_resp.json() + "\n").encode('utf-8')
        
        # 模拟流式输出
        if not is_final:
            await asyncio.sleep(0.1)

# 缓存统计端点
@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """
    获取缓存统计数据
    """
    stats = cache_manager.get_stats()
    return {"cache": stats}

# 清除缓存端点
@app.post("/api/v1/cache/clear")
async def clear_cache():
    """
    清除所有缓存
    """
    success = await cache_manager.clear_all()
    if success:
        return {"message": "所有缓存已清除"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清除缓存时出错"
        )

# 系统监控端点
@app.get("/api/v1/system/metrics")
async def get_system_metrics(history: bool = False, count: int = 60):
    """
    获取系统指标
    
    参数:
        history: 是否返回历史数据
        count: 返回历史数据的数量
    """
    if history:
        return {"metrics": system_monitor.get_metrics_history(count)}
    else:
        return {"metrics": system_monitor.get_latest_metrics()}

# 系统概况端点
@app.get("/api/v1/system/summary")
async def get_system_summary():
    """
    获取系统概况
    """
    return {"summary": system_monitor.get_summary()}

# 启动系统监控
@app.post("/api/v1/system/monitor/start")
async def start_system_monitor(interval: int = 60):
    """
    启动系统监控
    
    参数:
        interval: 监控采集间隔（秒）
    """
    system_monitor.interval = interval
    success = system_monitor.start()
    return {"success": success, "message": "系统监控已启动" if success else "系统监控启动失败"}

# 停止系统监控
@app.post("/api/v1/system/monitor/stop")
async def stop_system_monitor():
    """
    停止系统监控
    """
    success = system_monitor.stop()
    return {"success": success, "message": "系统监控已停止" if success else "系统监控停止失败"}

# 启动应用的入口
if __name__ == "__main__":
    # 使用uvicorn启动应用
    uvicorn.run(
        "api:app", 
        host=settings.HOST, 
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level="info"
    )