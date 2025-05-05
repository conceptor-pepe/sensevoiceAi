#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice Small模型ONNX部署API服务
"""

import os
import time
import json
import base64
import logging
from typing import Optional, List, Dict, Any, Union

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# 导入SenseVoice Small模型
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sensevoice-api')

# 创建FastAPI应用
app = FastAPI(
    title="SenseVoice API",
    description="SenseVoice Small模型ONNX部署API服务",
    version="1.0.0"
)

# 全局变量
MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "iic/SenseVoiceSmall")
GPU_DEVICE = os.environ.get("SENSEVOICE_GPU_DEVICE", "0")  # 指定GPU设备号
BATCH_SIZE = int(os.environ.get("SENSEVOICE_BATCH_SIZE", "1"))

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# 模型实例
model = None

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

@app.on_event("startup")
async def startup_event():
    """
    服务启动时加载模型
    """
    global model
    try:
        logger.info(f"正在加载SenseVoice Small模型，模型目录: {MODEL_DIR}, 使用GPU: {GPU_DEVICE}")
        model = SenseVoiceSmall(MODEL_DIR, batch_size=BATCH_SIZE, quantize=True)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

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
    global model
    
    start_time = time.time()
    
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "模型未加载成功", "time_cost": time.time() - start_time}
        )
    
    try:
        # 处理输入数据
        if audio_file is not None:
            # 保存上传的文件到临时目录
            temp_file = f"/tmp/sensevoice_{int(time.time())}_{audio_file.filename}"
            with open(temp_file, "wb") as f:
                f.write(await audio_file.read())
            
            audio_path = temp_file
        elif request_data is not None:
            # 解析JSON请求
            try:
                req_data = json.loads(request_data)
            except json.JSONDecodeError:
                req_data = {"audio_base64": request_data}
            
            if "audio_base64" not in req_data:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": "缺少audio_base64字段", 
                        "time_cost": time.time() - start_time
                    }
                )
            
            # 将base64解码为音频文件
            audio_base64 = req_data.get("audio_base64")
            language = req_data.get("language", language)
            use_itn = req_data.get("use_itn", use_itn)
            
            try:
                audio_data = base64.b64decode(audio_base64)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False, 
                        "message": f"Base64解码失败: {str(e)}", 
                        "time_cost": time.time() - start_time
                    }
                )
                
            # 保存到临时文件
            temp_file = f"/tmp/sensevoice_{int(time.time())}.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
                
            audio_path = temp_file
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False, 
                    "message": "请提供音频文件或base64编码的音频数据", 
                    "time_cost": time.time() - start_time
                }
            )
        
        # 执行模型推理
        logger.info(f"开始处理音频: {audio_path}, 语言: {language}, 使用ITN: {use_itn}")
        
        # 调用SenseVoice模型
        result = model([audio_path], language=language, use_itn=use_itn)
        
        # 处理结果
        if result and len(result) > 0:
            # 应用后处理
            processed_text = rich_transcription_postprocess(result[0])
            
            # 提取语言、情绪和事件信息
            language_tag = None
            emotion_tag = None
            event_tag = None
            
            # 尝试从结果中提取标签
            for tag in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]:
                if tag in result[0]:
                    language_tag = tag
                    break
                    
            for tag in ["<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", "<|FEARFUL|>", "<|DISGUSTED|>", "<|SURPRISED|>"]:
                if tag in result[0]:
                    emotion_tag = tag
                    break
                    
            for tag in ["<|BGM|>", "<|Speech|>", "<|Applause|>", "<|Laughter|>", "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>"]:
                if tag in result[0]:
                    event_tag = tag
                    break
            
            # 移除临时文件
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {str(e)}")
            
            time_cost = time.time() - start_time
            logger.info(f"处理完成，耗时: {time_cost:.2f}秒")
            
            return {
                "success": True,
                "message": "识别成功",
                "text": processed_text,
                "language": language_tag.replace("<|", "").replace("|>", "") if language_tag else None,
                "emotion": emotion_tag.replace("<|", "").replace("|>", "") if emotion_tag else None,
                "event": event_tag.replace("<|", "").replace("|>", "") if event_tag else None,
                "time_cost": time_cost
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False, 
                    "message": "模型处理失败，未返回结果", 
                    "time_cost": time.time() - start_time
                }
            )
            
    except Exception as e:
        logger.error(f"处理异常: {str(e)}")
        # 尝试清理临时文件
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "message": f"处理失败: {str(e)}", 
                "time_cost": time.time() - start_time
            }
        )

if __name__ == "__main__":
    # 服务启动配置
    HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")
    PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))
    
    # 启动服务
    uvicorn.run(app, host=HOST, port=PORT)
