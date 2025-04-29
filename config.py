import os
# 修改导入语句使用pydantic-settings包
from pydantic_settings import BaseSettings

# 配置类
class Settings(BaseSettings):
    """
    应用配置类
    """
    # 应用基础配置
    APP_NAME: str = "SenseVoice API"
    APP_VERSION: str = "1.0.0"
    
    # GPU配置
    DEVICE: str = "cuda:5"  # 指定使用第5号GPU
    
    # 模型配置
    MODEL_DIR: str = "iic/SenseVoiceSmall"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4  # Uvicorn工作进程数
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # 并发控制
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 600  # 秒
    
    # 流式处理配置
    CHUNK_SIZE: int = 4096  # 音频流块大小
    
    # 安全配置
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""
    
    # 缓存配置
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 秒
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 创建全局配置实例
settings = Settings() 