#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 测试脚本
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path

def test_health(url):
    """
    测试健康检查接口
    """
    try:
        response = requests.get(f"{url}")
        print("=== 健康检查接口测试 ===")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查测试失败: {str(e)}")
        return False

def test_recognize_upload(url, audio_file, language="auto", use_itn=True):
    """
    测试文件上传方式的识别接口
    """
    try:
        with open(audio_file, "rb") as f:
            start_time = time.time()
            response = requests.post(
                f"{url}/recognize",
                files={"audio_file": f},
                data={
                    "language": language,
                    "use_itn": str(use_itn).lower()
                }
            )
            elapsed = time.time() - start_time
            
            print("\n=== 文件上传方式识别接口测试 ===")
            print(f"音频文件: {audio_file}")
            print(f"请求耗时: {elapsed:.2f}秒")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
    except Exception as e:
        print(f"文件上传测试失败: {str(e)}")
        return False

def test_recognize_base64(url, audio_file, language="auto", use_itn=True):
    """
    测试Base64编码方式的识别接口
    """
    try:
        with open(audio_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            json_data = {
                "audio_base64": audio_base64,
                "language": language,
                "use_itn": use_itn
            }
            
            start_time = time.time()
            response = requests.post(
                f"{url}/recognize",
                data={"request_data": json.dumps(json_data)}
            )
            elapsed = time.time() - start_time
            
            print("\n=== Base64编码方式识别接口测试 ===")
            print(f"音频文件: {audio_file}")
            print(f"请求耗时: {elapsed:.2f}秒")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
    except Exception as e:
        print(f"Base64编码测试失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SenseVoice API测试工具")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API服务URL")
    parser.add_argument("--audio", type=str, required=True, help="测试音频文件路径")
    parser.add_argument("--language", type=str, default="auto", help="语言类型")
    parser.add_argument("--use_itn", type=bool, default=True, help="是否使用ITN")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件 {args.audio} 不存在")
        sys.exit(1)
    
    # 测试健康检查接口
    if not test_health(args.url):
        print("健康检查失败，API服务可能未正常运行")
        sys.exit(1)
    
    # 测试文件上传方式
    test_recognize_upload(args.url, args.audio, args.language, args.use_itn)
    
    # 测试Base64编码方式
    test_recognize_base64(args.url, args.audio, args.language, args.use_itn)
    
    print("\n所有测试完成")

if __name__ == "__main__":
    main() 