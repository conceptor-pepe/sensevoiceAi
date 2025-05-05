#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice WebSocket客户端示例
展示如何使用WebSocket接口进行流式语音识别
"""

import asyncio
import json
import wave
import sys
import websockets

# 配置信息
SERVER_URL = "ws://localhost:8000/ws/recognize"  # WebSocket服务地址
AUDIO_FILE = "test.wav"  # 音频文件
CHUNK_SIZE = 3200  # 每次发送的块大小 (16k采样率下约0.2秒)
LANGUAGE = "auto"  # 语言设置
USE_ITN = True  # 是否使用反向文本归一化

async def stream_audio():
    """流式处理音频文件"""
    print(f"连接到服务器: {SERVER_URL}")
    
    # 连接到WebSocket服务器
    async with websockets.connect(SERVER_URL) as websocket:
        # 发送配置信息
        config = {
            "language": LANGUAGE,
            "use_itn": USE_ITN
        }
        await websocket.send(json.dumps(config))
        
        # 接收准备就绪响应
        response = await websocket.recv()
        print(f"服务器响应: {response}")
        
        # 打开并读取音频文件
        with wave.open(AUDIO_FILE, 'rb') as wf:
            # 打印音频信息
            print(f"采样率: {wf.getframerate()}")
            print(f"声道数: {wf.getnchannels()}")
            print(f"样本宽度: {wf.getsampwidth()}")
            
            # 检查音频格式是否符合要求
            if wf.getframerate() != 16000 or wf.getnchannels() != 1:
                print("警告: 音频应为16kHz单声道WAV格式")
            
            # 循环读取音频数据并发送
            print("开始流式处理...")
            
            # 创建监听接收消息的任务
            receive_task = asyncio.create_task(receive_messages(websocket))
            
            # 读取并发送音频数据
            while True:
                data = wf.readframes(CHUNK_SIZE)
                if not data:
                    break
                
                # 发送音频数据
                await websocket.send(data)
                
                # 模拟实时发送的速度
                # 当前CHUNK_SIZE下约每0.2秒发送一次
                await asyncio.sleep(0.2)
            
            # 发送空数据表示结束
            await websocket.send(b"")
            
            # 等待接收任务完成
            await receive_task

async def receive_messages(websocket):
    """监听并接收WebSocket消息"""
    try:
        while True:
            # 接收识别结果
            message = await websocket.recv()
            result = json.loads(message)
            
            # 输出识别结果
            if "is_final" in result and result["is_final"]:
                print("\n最终识别结果:")
                print(f"文本: {result.get('text', '')}")
                print(f"语言: {result.get('language', 'unknown')}")
                print(f"情感: {result.get('emotion', 'unknown')}")
                print(f"事件: {result.get('event', 'unknown')}")
                print(f"总耗时: {result.get('time_cost', 0):.4f}秒")
                return  # 识别完成，结束接收
            else:
                print(f"\r当前识别: {result.get('text', '')}... ", end="", flush=True)
    except Exception as e:
        print(f"接收消息异常: {str(e)}")

def print_usage():
    """打印使用说明"""
    print(f"""
SenseVoice WebSocket客户端示例
用法: python {sys.argv[0]} [audio_file] [language] [use_itn]

参数:
  audio_file  - WAV音频文件路径 (默认: {AUDIO_FILE})
  language    - 识别语言代码 (默认: {LANGUAGE})
  use_itn     - 是否使用反向文本归一化 (true/false, 默认: {USE_ITN})

示例:
  python {sys.argv[0]} test.wav zh true

注意: 音频应为16kHz单声道WAV格式
    """)

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print_usage()
            sys.exit(0)
        AUDIO_FILE = sys.argv[1]
    
    if len(sys.argv) > 2:
        LANGUAGE = sys.argv[2]
    
    if len(sys.argv) > 3:
        USE_ITN = sys.argv[3].lower() == "true"
    
    # 运行客户端
    asyncio.run(stream_audio()) 