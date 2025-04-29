import hashlib
import logging
import os
import time
from typing import Dict, Optional, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("utils")

def generate_hash(data: bytes) -> str:
    """
    生成数据的哈希值
    
    参数:
        data: 要哈希的字节数据
        
    返回:
        哈希字符串
    """
    hasher = hashlib.md5()
    hasher.update(data)
    return hasher.hexdigest()

def measure_performance(func):
    """
    性能测量装饰器
    
    参数:
        func: 要测量的函数
        
    返回:
        装饰后的函数
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"函数 {func.__name__} 执行耗时: {elapsed_time:.4f} 秒")
        return result
    return wrapper

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    参数:
        size_bytes: 字节大小
        
    返回:
        格式化后的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def validate_audio_format(file_content_type: str, filename: Optional[str] = None) -> bool:
    """
    验证音频格式是否支持
    
    参数:
        file_content_type: 文件的MIME类型
        filename: 文件名（用于检查扩展名）
        
    返回:
        是否为支持的格式
    """
    # 支持的MIME类型
    valid_content_types = [
        "audio/wav", 
        "audio/x-wav",
        "audio/mp3", 
        "audio/mpeg",
        "audio/ogg", 
        "audio/webm",
        "application/octet-stream"  # 某些客户端可能使用这种通用类型
    ]
    
    # 支持的文件扩展名
    valid_extensions = [".wav", ".mp3", ".ogg", ".webm"]
    
    # 检查MIME类型
    is_valid_content_type = any(valid_type in file_content_type for valid_type in valid_content_types)
    
    # 如果MIME类型不明确，尝试检查文件扩展名
    if not is_valid_content_type and filename:
        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in valid_extensions
    
    return is_valid_content_type

def filter_sensitive_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤敏感信息
    
    参数:
        data: 包含可能敏感信息的字典
        
    返回:
        过滤后的字典
    """
    sensitive_keys = ["password", "api_key", "secret", "token"]
    filtered_data = data.copy()
    
    for key in filtered_data:
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            filtered_data[key] = "********"
    
    return filtered_data

def get_error_message(exception: Exception) -> str:
    """
    获取格式化的错误消息
    
    参数:
        exception: 异常对象
        
    返回:
        格式化的错误消息
    """
    error_type = type(exception).__name__
    error_message = str(exception)
    return f"{error_type}: {error_message}" 