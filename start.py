#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SenseVoice 主入口脚本
提供命令行界面启动SenseVoice服务，支持不同的运行模式和配置
"""

import os
import sys
import argparse
import uvicorn
import time
import logging
from pathlib import Path

# 确保当前目录在Python搜索路径中，以便导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入配置和日志模块
import config
from logger import logger, ensure_log_dir

def show_banner():
    """显示程序横幅"""
    banner = """
    ███████╗███████╗███╗   ██╗███████╗███████╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
    ██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝██║   ██║██╔═══██╗██║██╔════╝██╔════╝
    ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  ██║   ██║██║   ██║██║██║     █████╗  
    ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
    ███████║███████╗██║ ╚████║███████║███████╗ ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
    ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
    """
    print(banner)
    print(f"  SenseVoice API 服务 v1.0.0")
    print(f"  日志目录: {config.LOG_DIR}")
    print(f"  模型目录: {config.MODEL_DIR}")
    print(f"  配置文件: {os.path.abspath('config.py')}")
    if os.path.exists('local_config.py'):
        print(f"  本地配置: {os.path.abspath('local_config.py')}")
    print("\n" + "="*80 + "\n")


def setup_environment():
    """设置运行环境，确保必要的目录存在"""
    # 确保日志目录存在
    ensure_log_dir()
    
    # 确保测试目录存在
    test_dir = Path(config.TEST_DIR)
    if not test_dir.exists():
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建测试目录: {test_dir}")
        except Exception as e:
            logger.error(f"无法创建测试目录 {test_dir}: {str(e)}")


def run_server(host=None, port=None, reload=False):
    """
    启动API服务器
    
    Args:
        host (str): 服务器主机地址，默认使用配置文件中的值
        port (int): 服务器端口，默认使用配置文件中的值
        reload (bool): 是否启用热重载，默认为False
    """
    # 使用命令行参数或配置文件中的值
    host = host or config.HOST
    port = port or config.PORT
    
    logger.info(f"启动SenseVoice API服务，监听地址: {host}:{port}")
    
    # 构建uvicorn配置
    uvicorn_config = {
        "app": "api:app",            # 指定FastAPI应用实例
        "host": host,                # 监听地址
        "port": port,                # 监听端口
        "reload": reload,            # 是否启用热重载
        "log_level": "info",         # uvicorn日志级别
        "access_log": True,          # 是否记录访问日志
        "workers": 1                 # 工作进程数量
    }
    
    # 启动服务器
    uvicorn.run(**uvicorn_config)


def run_test():
    """运行测试"""
    logger.info("启动测试模式")
    
    try:
        import test_api
        test_api.main()
    except ImportError:
        logger.error("无法导入测试模块 test_api.py")
        print("错误: 无法导入测试模块。请确保 test_api.py 文件存在。")
    except Exception as e:
        logger.error(f"测试运行时出错: {str(e)}")
        print(f"测试失败: {str(e)}")


# 给出执行例子
# python3 start.py --host 0.0.0.0 --port 8000 --reload --test --debug
def main():
    """主函数，解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description="SenseVoice API 服务")
    
    # 定义命令行参数
    parser.add_argument("--host", help="服务器主机地址 (默认: 配置文件中的HOST)")
    parser.add_argument("--port", type=int, help="服务器端口 (默认: 配置文件中的PORT)")
    parser.add_argument("--reload", action="store_true", help="启用热重载 (仅开发环境)")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 显示程序横幅
    show_banner()
    
    # 设置运行环境
    setup_environment()
    
    # 根据命令行参数执行不同操作
    if args.debug:
        # 调试模式下修改日志级别
        logger.setLevel(logging.DEBUG)
        logger.info("启用调试模式")
    
    if args.test:
        # 测试模式
        run_test()
    else:
        # 服务器模式
        run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    # 记录启动时间
    start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("收到中断信号，程序退出")
        print("\n程序已停止")
    except Exception as e:
        logger.error(f"程序运行时出错: {str(e)}")
        print(f"错误: {str(e)}")
    finally:
        # 记录运行时间
        run_time = time.time() - start_time
        logger.info(f"程序运行时间: {run_time:.2f}秒") 