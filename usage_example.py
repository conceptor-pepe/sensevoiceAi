#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage_example.py - SenseVoice API 使用示例

该示例程序演示了如何使用SenseVoice API进行语音识别的多种方式：
1. 直接使用模型进行本地识别
2. 通过HTTP API进行批量识别
3. 通过WebSocket进行流式识别

作者: SenseVoice团队
日期: 2023-09-15
"""

import os
import sys
import time
import json
import wave
import asyncio
import argparse
import requests
import numpy as np
import websockets

# 本地模式的导入
try:
    from model import SenseVoiceSmall
except ImportError:
    print("警告: 未找到本地模型模块，仅支持API调用模式")

class SenseVoiceClient:
    """
    SenseVoice 客户端类
    支持本地模型调用和API远程调用两种方式
    """
    
    def __init__(self, mode="local", api_url="http://localhost:8000", model_dir=None):
        """
        初始化SenseVoice客户端
        
        参数:
            mode: 运行模式，'local'表示本地模型，'api'表示远程API
            api_url: API服务器地址
            model_dir: 模型目录，本地模式使用
        """
        self.mode = mode
        self.api_url = api_url.rstrip('/')
        self.model = None
        
        # 本地模式初始化
        if mode == "local":
            try:
                print(f"正在加载本地模型: {model_dir}")
                self.model = SenseVoiceSmall(model_dir=model_dir)
                print(f"模型加载完成，后端: {self.model.backend}")
            except Exception as e:
                print(f"错误: 无法加载本地模型: {str(e)}")
                print("将回退到API模式")
                self.mode = "api"
    
    def recognize_file(self, audio_file, language="auto"):
        """
        识别音频文件
        
        参数:
            audio_file: 音频文件路径
            language: 语言选择，auto表示自动检测
            
        返回:
            识别结果
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
            
        start_time = time.time()
        
        if self.mode == "local" and self.model is not None:
            # 本地模式
            print(f"使用本地模型识别: {audio_file}")
            result = self.model.infer(audio_file, language=language)
        else:
            # API模式
            print(f"使用远程API识别: {audio_file}")
            result = self._call_api(audio_file, language)
            
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 添加处理时间
        if isinstance(result, dict) and "processing_time" not in result:
            result["processing_time"] = elapsed
            
        return result
    
    def _call_api(self, audio_file, language):
        """
        调用REST API进行识别
        
        参数:
            audio_file: 音频文件路径
            language: 语言选择
            
        返回:
            识别结果字典
        """
        url = f"{self.api_url}/api/v1/asr"
        
        try:
            with open(audio_file, "rb") as f:
                files = {"audio_file": (os.path.basename(audio_file), f, "audio/wav")}
                data = {"language": language}
                
                response = requests.post(url, files=files, data=data)
                
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API请求失败: HTTP {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f", {error_details.get('detail', '')}"
                except:
                    error_msg += f", {response.text}"
                raise Exception(error_msg)
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return {"error": str(e)}
            
    async def stream_recognize(self, audio_file, language="auto", chunk_size=4096, sample_rate=16000):
        """
        流式识别音频文件
        
        参数:
            audio_file: 音频文件路径
            language: 语言选择
            chunk_size: 音频分块大小
            sample_rate: 音频采样率
            
        返回:
            最终识别结果
        """
        if self.mode == "local":
            print("警告: 本地模式暂不支持流式识别，将使用API模式")
            
        uri = f"ws://{self.api_url.split('://')[-1]}/api/v1/asr/stream"
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"已连接WebSocket: {uri}")
                
                # 发送初始配置
                config = {
                    "language": language,
                    "sample_rate": sample_rate
                }
                await websocket.send(json.dumps(config))
                print(f"已发送配置: {config}")
                
                # 分块发送音频
                with wave.open(audio_file, "rb") as wav:
                    print(f"开始发送音频: {audio_file}")
                    frames_sent = 0
                    
                    while True:
                        data = wav.readframes(chunk_size)
                        if not data:
                            break
                            
                        await websocket.send(data)
                        frames_sent += len(data)
                        print(f"已发送 {frames_sent} 字节数据")
                        
                        # 接收中间结果
                        try:
                            # 尝试接收但不阻塞
                            interim_result = await asyncio.wait_for(websocket.recv(), 0.01)
                            interim_result = json.loads(interim_result)
                            print(f"中间结果: {interim_result.get('text', '')}")
                        except asyncio.TimeoutError:
                            pass
                    
                    # 发送结束标记
                    await websocket.send(json.dumps({"eof": True}))
                    print("音频发送完成，已发送EOF标记")
                
                # 接收最终结果
                final_result = None
                while True:
                    result = await websocket.recv()
                    result = json.loads(result)
                    
                    if result.get("final"):
                        final_result = result
                        print(f"最终结果: {result}")
                        break
                    else:
                        print(f"中间结果: {result}")
                
                return final_result
                
        except Exception as e:
            print(f"流式识别错误: {str(e)}")
            return {"error": str(e)}
    
    def get_server_info(self):
        """
        获取服务器信息
        
        返回:
            服务器信息字典
        """
        if self.mode == "local":
            if self.model:
                return self.model.get_model_info()
            else:
                return {"error": "本地模型未加载"}
        else:
            url = f"{self.api_url}/api/v1/info"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
            except Exception as e:
                return {"error": str(e)}

def create_test_audio(output_file="test.wav", duration=3, sample_rate=16000):
    """
    创建测试用的音频文件（纯音调）
    
    参数:
        output_file: 输出文件路径
        duration: 音频时长（秒）
        sample_rate: 采样率
    """
    # 创建一个音调
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz 音调
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 保存为WAV文件
    with wave.open(output_file, 'w') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位
        wf.setframerate(sample_rate)
        wf.writeframes((tone * 32767).astype(np.int16).tobytes())
    
    print(f"已创建测试音频: {output_file}")
    return output_file

async def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="SenseVoice API 使用示例")
    parser.add_argument("--mode", choices=["local", "api"], default="api",
                        help="运行模式: local使用本地模型, api调用远程服务")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API服务器地址 (默认: http://localhost:8000)")
    parser.add_argument("--model-dir", default="iic/SenseVoiceSmall",
                        help="本地模型目录 (默认: iic/SenseVoiceSmall)")
    parser.add_argument("--audio-file", default="",
                        help="要识别的音频文件路径，若不提供则创建测试音频")
    parser.add_argument("--language", default="auto",
                        choices=["auto", "zh", "en"], help="语言选择 (默认: auto)")
    parser.add_argument("--stream", action="store_true",
                        help="使用流式识别")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = SenseVoiceClient(mode=args.mode, api_url=args.api_url, model_dir=args.model_dir)
    
    # 检查或创建音频文件
    audio_file = args.audio_file
    if not audio_file:
        audio_file = create_test_audio()
    
    # 获取服务器信息
    server_info = client.get_server_info()
    print(f"\n服务器信息:\n{json.dumps(server_info, indent=2, ensure_ascii=False)}")
    
    # 执行识别
    if args.stream:
        print("\n执行流式识别...")
        result = await client.stream_recognize(audio_file, language=args.language)
    else:
        print("\n执行批量识别...")
        result = client.recognize_file(audio_file, language=args.language)
    
    # 打印结果
    print(f"\n识别结果:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    
    if "results" in result and result["results"]:
        print(f"\n识别文本: {result['results'][0]['text']}")
    elif "text" in result:
        print(f"\n识别文本: {result['text']}")

if __name__ == "__main__":
    asyncio.run(main()) 