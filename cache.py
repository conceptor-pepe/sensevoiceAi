import redis
import json
import hashlib
import logging
import time
import traceback
from typing import Any, Dict, Optional, Union
from config import settings
from performance import measure_performance

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cache_manager")

class CacheManager:
    """
    缓存管理器类，负责结果缓存
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化缓存管理器
        """
        if not self._initialized:
            self.enabled = settings.CACHE_ENABLED
            self.ttl = settings.CACHE_TTL
            
            # 性能统计
            self._hit_count = 0  # 缓存命中次数
            self._miss_count = 0  # 缓存未命中次数
            self._set_count = 0  # 缓存设置次数
            self._error_count = 0  # 错误次数
            
            if self.enabled:
                try:
                    start_time = time.time()
                    logger.info("正在连接Redis缓存...")
                    
                    self.redis = redis.Redis(
                        host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT,
                        db=settings.REDIS_DB,
                        password=settings.REDIS_PASSWORD,
                        decode_responses=True,
                        socket_timeout=5.0,  # 添加超时设置
                        socket_connect_timeout=5.0,
                        retry_on_timeout=True
                    )
                    
                    self.redis.ping()  # 测试连接
                    connect_time = time.time() - start_time
                    logger.info(f"Redis缓存连接成功，耗时: {connect_time:.4f}秒")
                except Exception as e:
                    self.enabled = False
                    logger.error(f"Redis连接失败，缓存将被禁用: {str(e)}")
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            self._initialized = True
            logger.info(f"缓存管理器初始化完成，状态: {'已启用' if self.enabled else '已禁用'}")
    
    def _generate_key(self, audio_data: bytes, language: str) -> str:
        """
        生成缓存键
        
        参数:
            audio_data: 音频数据
            language: 语言
            
        返回:
            缓存键字符串
        """
        # 使用音频数据的MD5哈希值和语言作为缓存键
        start_time = time.time()
        hasher = hashlib.md5()
        hasher.update(audio_data)
        audio_hash = hasher.hexdigest()
        cache_key = f"sensevoice:transcription:{audio_hash}:{language}"
        
        hash_time = time.time() - start_time
        if hash_time > 0.1:  # 如果哈希计算超过100毫秒，记录日志
            logger.warning(f"缓存键生成耗时较长: {hash_time:.4f}秒，数据大小: {len(audio_data)} 字节")
        
        return cache_key
    
    @measure_performance(lambda self, audio_data, language: 
        {"data_size": len(audio_data), "language": language})
    async def get(self, audio_data: bytes, language: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取结果
        
        参数:
            audio_data: 音频数据
            language: 语言
            
        返回:
            缓存的结果或None
        """
        if not self.enabled:
            self._miss_count += 1
            return None
        
        key = self._generate_key(audio_data, language)
        
        try:
            start_time = time.time()
            logger.info(f"尝试从缓存获取: {key}")
            
            data = self.redis.get(key)
            get_time = time.time() - start_time
            
            if data:
                self._hit_count += 1
                logger.info(f"缓存命中: {key}, 获取耗时: {get_time:.4f}秒")
                
                # 尝试解析JSON
                try:
                    parsed_data = json.loads(data)
                    return parsed_data
                except json.JSONDecodeError as e:
                    logger.error(f"解析缓存数据失败: {key}, 错误: {str(e)}")
                    self._error_count += 1
                    return None
            else:
                self._miss_count += 1
                logger.info(f"缓存未命中: {key}, 查询耗时: {get_time:.4f}秒")
                return None
        except Exception as e:
            self._error_count += 1
            self._miss_count += 1
            logger.error(f"从缓存获取数据失败: {key}, 错误: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    @measure_performance(lambda self, audio_data, language, result: 
        {"data_size": len(audio_data), "language": language, "result_size": len(str(result))})
    async def set(self, audio_data: bytes, language: str, result: Dict[str, Any]) -> bool:
        """
        将结果存入缓存
        
        参数:
            audio_data: 音频数据
            language: 语言
            result: 要缓存的结果
            
        返回:
            是否成功
        """
        if not self.enabled:
            return False
        
        key = self._generate_key(audio_data, language)
        
        try:
            # 转换为JSON字符串
            start_time = time.time()
            json_data = json.dumps(result)
            json_size = len(json_data)
            
            # 存储到Redis
            logger.info(f"正在缓存结果: {key}, 大小: {json_size} 字节")
            self.redis.setex(key, self.ttl, json_data)
            
            set_time = time.time() - start_time
            logger.info(f"结果已缓存: {key}, 耗时: {set_time:.4f}秒")
            
            self._set_count += 1
            return True
        except Exception as e:
            self._error_count += 1
            logger.error(f"缓存结果失败: {key}, 错误: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    @measure_performance
    async def delete(self, audio_data: bytes, language: str) -> bool:
        """
        从缓存中删除结果
        
        参数:
            audio_data: 音频数据
            language: 语言
            
        返回:
            是否成功
        """
        if not self.enabled:
            return False
        
        key = self._generate_key(audio_data, language)
        
        try:
            start_time = time.time()
            logger.info(f"正在删除缓存: {key}")
            
            result = self.redis.delete(key)
            delete_time = time.time() - start_time
            
            if result > 0:
                logger.info(f"缓存已删除: {key}, 耗时: {delete_time:.4f}秒")
            else:
                logger.info(f"缓存键不存在: {key}, 耗时: {delete_time:.4f}秒")
            
            return True
        except Exception as e:
            self._error_count += 1
            logger.error(f"删除缓存失败: {key}, 错误: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    @measure_performance
    async def clear_all(self) -> bool:
        """
        清除所有缓存
        
        返回:
            是否成功
        """
        if not self.enabled:
            return False
        
        try:
            start_time = time.time()
            logger.info("正在清除所有缓存...")
            
            keys = self.redis.keys("sensevoice:transcription:*")
            keys_count = len(keys)
            
            if keys:
                self.redis.delete(*keys)
                clear_time = time.time() - start_time
                logger.info(f"已清除所有缓存，共{keys_count}项，耗时: {clear_time:.4f}秒")
            else:
                logger.info("没有找到需要清除的缓存")
            
            return True
        except Exception as e:
            self._error_count += 1
            logger.error(f"清除所有缓存失败: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        返回:
            缓存统计信息字典
        """
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / max(1, total_requests)) * 100
        
        stats = {
            "enabled": self.enabled,
            "total_requests": total_requests,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "set_count": self._set_count,
            "error_count": self._error_count,
            "hit_rate": f"{hit_rate:.2f}%"
        }
        
        # 如果缓存已启用，尝试获取Redis信息
        if self.enabled:
            try:
                info = self.redis.info()
                stats["redis_info"] = {
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "uptime_days": info.get("uptime_in_days", 0)
                }
            except Exception as e:
                logger.error(f"获取Redis信息失败: {str(e)}")
                stats["redis_info"] = {"error": str(e)}
        
        return stats


# 创建全局缓存管理器实例
cache_manager = CacheManager() 