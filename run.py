import uvicorn
import os
import argparse
import logging
import time
from config import settings
from logging_config import configure_logging
from monitoring import system_monitor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("runner")

#python run.py --host 0.0.0.0 --port 8000 --workers 4 --device cuda:5
def main():
    """
    主程序入口，处理命令行参数并启动服务
    """
    start_time = time.time()
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='SenseVoice API服务')
    parser.add_argument('--host', type=str, default=settings.HOST,
                        help='监听主机名 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=settings.PORT,
                        help='监听端口 (默认: 8000)')
    parser.add_argument('--workers', type=int, default=settings.WORKERS,
                        help='工作进程数 (默认: 4)')
    parser.add_argument('--device', type=str, default=settings.DEVICE,
                        help='CUDA设备 (默认: cuda:5)')
    parser.add_argument('--model-dir', type=str, default=settings.MODEL_DIR,
                        help='模型目录 (默认: iic/SenseVoiceSmall)')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='日志级别 (默认: info)')
    parser.add_argument('--reload', action='store_true',
                        help='开发模式下启用热重载')
    parser.add_argument('--monitor-interval', type=int, default=60,
                        help='系统监控间隔秒数 (默认: 60)')
    parser.add_argument('--no-monitor', action='store_true',
                        help='禁用系统监控')
    
    args = parser.parse_args()
    
    # 配置日志系统
    log_level = getattr(logging, args.log_level.upper())
    log_config = configure_logging(log_level)
    
    # 设置环境变量
    os.environ["SENSEVOICE_DEVICE"] = args.device
    
    # 启动系统监控
    if not args.no_monitor:
        logger.info(f"正在启动系统监控，间隔: {args.monitor_interval}秒")
        system_monitor.interval = args.monitor_interval
        system_monitor.start()
    else:
        logger.info("系统监控已禁用")
    
    # 打印启动信息
    logger.info(f"启动 {settings.APP_NAME} 版本 {settings.APP_VERSION}")
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"工作进程数: {args.workers}")
    logger.info(f"GPU设备: {args.device}")
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"日志配置: 级别={log_config['log_level']}, 文件={log_config['log_file']}")
    
    # 启动服务
    try:
        uvicorn.run(
            "api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload
        )
    except Exception as e:
        logger.error(f"启动服务时出错: {str(e)}")
        raise
    finally:
        # 如果启动了监控，在结束时停止监控
        if not args.no_monitor:
            logger.info("正在停止系统监控...")
            system_monitor.stop()
        
        end_time = time.time()
        uptime = end_time - start_time
        logger.info(f"服务运行时间: {uptime:.2f} 秒")

if __name__ == "__main__":
    main()