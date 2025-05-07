#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice API测试脚本
"""
import requests
import time
import sys
import os
from pathlib import Path

def test_status():
    """测试状态端点"""
    print("测试 /status 端点...")
    try:
        response = requests.get("http://localhost:8000/status")
        response.raise_for_status()
        data = response.json()
        print(f"状态: {data['status']}")
        print(f"版本: {data['version']}")
        print(f"GPU状态: {data['gpu_status']}")
        print("✓ 状态端点测试通过!")
        return True
    except Exception as e:
        print(f"✗ 状态端点测试失败: {str(e)}")
        return False

def test_transcribe(audio_file):
    """测试转写端点"""
    if not os.path.exists(audio_file):
        print(f"✗ 测试文件不存在: {audio_file}")
        return False
    
    print(f"测试 /transcribe 端点 (文件: {audio_file})...")
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (os.path.basename(audio_file), f)}
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/transcribe",
                files=files
            )
        
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time
        
        print(f"识别结果: {data['text']}")
        print(f"处理时间: {data['processing_time']:.3f}秒")
        print(f"请求总时间: {elapsed:.3f}秒")
        print("✓ 转写端点测试通过!")
        return True
    except Exception as e:
        print(f"✗ 转写端点测试失败: {str(e)}")
        return False

def test_batch_transcribe(audio_file):
    """测试批量转写端点"""
    if not os.path.exists(audio_file):
        print(f"✗ 测试文件不存在: {audio_file}")
        return False
    
    print(f"测试 /api/v1/asr 端点 (文件: {audio_file})...")
    try:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
            
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/asr",
            files={"files": audio_data},
            data={"keys": "test_file", "lang": "auto"}
        )
        
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time
        
        print(f"识别结果: {data}")
        print(f"请求总时间: {elapsed:.3f}秒")
        print("✓ 批量转写端点测试通过!")
        return True
    except Exception as e:
        print(f"✗ 批量转写端点测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not audio_file:
        print("用法: python test_api.py <音频文件路径>")
        sys.exit(1)
    
    print("开始测试SenseVoice API...")
    
    success = test_status()
    
    if success:
        success = test_transcribe(audio_file)
    
    if success:
        success = test_batch_transcribe(audio_file)
    
    if success:
        print("所有测试通过! API运行正常。")
    else:
        print("部分测试失败，请检查错误信息。") 