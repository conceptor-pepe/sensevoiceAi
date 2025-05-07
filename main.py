#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口 - 启动FastAPI服务
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

import config
from logger import logger
from model_manager import ModelManager

def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的命令行参数对象
    """
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
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """
    设置运行环境
    
    参数:
        args: 命令行参数对象
    """
    # 设置GPU设备
    if args.gpu != config.GPU_DEVICE_ID:
        config.GPU_DEVICE_ID = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 设置调试环境变量
    if args.debug:
        os.environ["DEBUG"] = "true"
        logger.info("调试模式已启用")
    
    # 检查日志目录
    check_log_directory()

def check_log_directory():
    """
    检查日志目录是否存在并可写
    如果不存在或没有写入权限，尝试使用stderr代替
    """
    if not config.LOG_DIR.exists():
        try:
            # 尝试创建日志目录
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"已创建日志目录: {config.LOG_DIR}")
        except PermissionError:
            logger.warning(f"无权限创建日志目录: {config.LOG_DIR}")
            logger.warning("请确保以足够权限运行或使用 'sudo mkdir -p /var/log/sensevoice && sudo chmod 777 /var/log/sensevoice'")
            if os.getenv("USE_STDERR", "false").lower() in ("true", "1", "yes"):
                logger.warning("将使用标准错误输出代替日志文件")
            else:
                logger.critical("无法访问日志目录，请以管理员权限运行")
                sys.exit(1)
    elif not os.access(config.LOG_DIR, os.W_OK):
        logger.warning(f"日志目录 {config.LOG_DIR} 存在但无写入权限")
        if os.getenv("USE_STDERR", "false").lower() in ("true", "1", "yes"):
            logger.warning("将使用标准错误输出代替日志文件")
        else:
            logger.critical("无法写入日志目录，请以管理员权限运行或修改权限")
            sys.exit(1)
    else:
        logger.info(f"日志目录就绪: {config.LOG_DIR}")

def main():
    """
    主函数 - 服务入口点
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置环境
    setup_environment(args)
    
    try:
        # 预初始化模型（这将确保在服务启动前模型已加载）
        logger.info("正在预初始化语音识别模型...")
        model_manager = ModelManager()
        logger.info("模型初始化完成，准备启动API服务")
        
        # 启动API服务
        logger.info(f"启动API服务: {args.host}:{args.port}，工作进程数: {args.workers}")
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