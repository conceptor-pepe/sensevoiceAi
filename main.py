#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口 - 启动FastAPI服务
"""
import os
import sys
import argparse
import uvicorn

import config
from logger import logger
from model_manager import ModelManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SenseVoice ASR API服务")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=config.API_HOST,
        help=f"API服务主机地址 (默认: {config.API_HOST})"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=config.API_PORT,
        help=f"API服务端口 (默认: {config.API_PORT})"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=config.API_WORKERS,
        help=f"Worker进程数 (默认: {config.API_WORKERS})"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=config.GPU_DEVICE_ID,
        help=f"指定GPU设备ID (默认: {config.GPU_DEVICE_ID})"
    )
    
    parser.add_argument(
        "--cache", 
        type=bool, 
        default=config.CACHE_ENABLED,
        help=f"是否启用缓存 (默认: {config.CACHE_ENABLED})"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU设备
    if args.gpu != config.GPU_DEVICE_ID:
        config.GPU_DEVICE_ID = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 设置缓存
    config.CACHE_ENABLED = args.cache
    
    try:
        # 预初始化模型（这将确保在服务启动前模型已加载）
        logger.info("正在预初始化语音识别模型...")
        model_manager = ModelManager()
        logger.info("模型初始化完成，准备启动API服务")
        
        # 启动API服务
        uvicorn.run(
            "api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="debug" if args.debug else "info",
            reload=args.debug,
            access_log=True
        )
    except Exception as e:
        logger.critical(f"服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 