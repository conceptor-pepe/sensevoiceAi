import os
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any
from config import settings

# 创建日志目录
def ensure_log_dir():
    """
    确保日志目录存在
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# 日志格式配置
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] [%(thread)d] - %(message)s'

# 日志配置
def configure_logging(log_level=logging.INFO):
    """
    配置日志系统
    
    参数:
        log_level: 日志级别
    """
    log_dir = ensure_log_dir()
    
    # 生成日志文件名，包含日期
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'{settings.APP_NAME.lower().replace(" ", "_")}_{date_str}.log')
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(DEFAULT_FORMAT)
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # 创建文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(DETAILED_FORMAT)
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # 配置特定模块的日志级别
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    
    logging.info(f"日志系统初始化完成，级别: {logging.getLevelName(log_level)}, 日志文件: {log_file}")
    
    return {
        'log_file': log_file,
        'log_level': logging.getLevelName(log_level)
    }

# 性能日志记录器
class PerformanceLogger:
    """
    性能日志记录器类
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceLogger, cls).__new__(cls)
            # 创建专用的性能日志记录器
            log_dir = ensure_log_dir()
            perf_logger = logging.getLogger('performance')
            
            # 配置性能日志文件
            date_str = datetime.now().strftime('%Y-%m-%d')
            perf_log_file = os.path.join(log_dir, f'performance_{date_str}.log')
            
            # 性能日志处理器
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            perf_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            perf_handler.setFormatter(perf_format)
            
            # 清除原有处理器
            for handler in perf_logger.handlers[:]:
                perf_logger.removeHandler(handler)
                
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
            perf_logger.propagate = False  # 避免重复记录
            
            cls._instance.logger = perf_logger
            cls._instance.log_file = perf_log_file
            
        return cls._instance
    
    def log_performance(self, function_name: str, execution_time: float, metadata: Dict[str, Any] = None):
        """
        记录性能日志
        
        参数:
            function_name: 函数名
            execution_time: 执行时间（秒）
            metadata: 元数据
        """
        metadata_str = ''
        if metadata:
            metadata_str = ' ' + ' '.join([f'{k}={v}' for k, v in metadata.items()])
        
        self.logger.info(f"性能记录: {function_name} - 耗时={execution_time:.4f}秒{metadata_str}")

# 错误日志记录器
class ErrorLogger:
    """
    错误日志记录器类
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorLogger, cls).__new__(cls)
            # 创建专用的错误日志记录器
            log_dir = ensure_log_dir()
            error_logger = logging.getLogger('errors')
            
            # 配置错误日志文件
            date_str = datetime.now().strftime('%Y-%m-%d')
            error_log_file = os.path.join(log_dir, f'errors_{date_str}.log')
            
            # 错误日志处理器
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
            )
            error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
            error_handler.setFormatter(error_format)
            error_handler.setLevel(logging.ERROR)
            
            # 清除原有处理器
            for handler in error_logger.handlers[:]:
                error_logger.removeHandler(handler)
                
            error_logger.addHandler(error_handler)
            error_logger.setLevel(logging.ERROR)
            # 允许传播到根日志器，使错误同时记录到主日志和错误日志
            error_logger.propagate = True
            
            cls._instance.logger = error_logger
            cls._instance.log_file = error_log_file
            
        return cls._instance
    
    def log_error(self, message: str, error: Exception = None, context: Dict[str, Any] = None):
        """
        记录错误日志
        
        参数:
            message: 错误消息
            error: 异常对象
            context: 上下文信息
        """
        log_message = message
        
        if error:
            log_message += f" - 异常: {type(error).__name__}: {str(error)}"
        
        if context:
            context_str = ' '.join([f'{k}={v}' for k, v in context.items()])
            log_message += f" - 上下文: {context_str}"
        
        self.logger.error(log_message)

# 创建实例
performance_logger = PerformanceLogger()
error_logger = ErrorLogger() 