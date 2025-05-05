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
import re
from typing import Optional, List, Dict, Any, Union, Annotated
from io import BytesIO

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

# 用于移除标签的正则表达式
regex = r"<\|.*?\|>"

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
    global model
    
    start_time = time.time()
    
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"result": [], "message": "模型未加载成功", "time_cost": time.time() - start_time}
        )
    
    try:
        # 处理输入数据
        audio_paths = []
        for file in files:
            # 保存上传的文件到临时目录
            temp_file = f"/tmp/sensevoice_{int(time.time())}_{file.filename}"
            content = await file.read()
            with open(temp_file, "wb") as f:
                f.write(content)
            audio_paths.append(temp_file)
        
        # 处理键值
        if keys == "":
            key_list = ["wav_file_tmp_name"]
        else:
            key_list = keys.split(",")
        
        # 检查语言设置
        if lang == "":
            lang = "auto"
        
        # 执行模型推理
        logger.info(f"开始处理音频: {audio_paths}, 语言: {lang}, 使用ITN: {use_itn}")
        
        # 调用SenseVoice模型
        results = model(audio_paths, language=lang, use_itn=use_itn)
        
        # 处理结果
        if results and len(results) > 0:
            # 格式化输出结果
            output_results = []
            
            for i, result in enumerate(results):
                # 获取当前文件对应的key
                key = key_list[i] if i < len(key_list) else f"unknown_{i}"
                
                # 提取标签信息
                language_tag = None
                emotion_tag = None
                event_tag = None
                
                for tag in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>"]:
                    if tag in result:
                        language_tag = tag.replace("<|", "").replace("|>", "")
                        break
                        
                for tag in ["<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", "<|FEARFUL|>", "<|DISGUSTED|>", "<|SURPRISED|>"]:
                    if tag in result:
                        emotion_tag = tag.replace("<|", "").replace("|>", "")
                        break
                        
                for tag in ["<|BGM|>", "<|Speech|>", "<|Applause|>", "<|Laughter|>", "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>"]:
                    if tag in result:
                        event_tag = tag.replace("<|", "").replace("|>", "")
                        break
                
                # 生成不同处理级别的文本
                raw_text = result
                clean_text = re.sub(regex, "", result, 0, re.MULTILINE)
                processed_text = rich_transcription_postprocess(result)
                
                # 添加到结果列表
                output_results.append({
                    "key": key,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "text": processed_text,
                    "language": language_tag,
                    "emotion": emotion_tag,
                    "event": event_tag
                })
            
            # 清理临时文件
            for path in audio_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")
            
            time_cost = time.time() - start_time
            logger.info(f"处理完成，耗时: {time_cost:.2f}秒")
            
            return {"result": output_results}
        else:
            return {"result": []}
            
    except Exception as e:
        logger.error(f"处理异常: {str(e)}")
        # 尝试清理临时文件
        try:
            for path in audio_paths:
                if os.path.exists(path):
                    os.remove(path)
        except:
            pass
            
        return JSONResponse(
            status_code=500,
            content={"result": [], "message": f"处理失败: {str(e)}"}
        )

if __name__ == "__main__":
    # 服务启动配置
    HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")
    PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))
    
    # 启动服务
    uvicorn.run(app, host=HOST, port=PORT)
