import time
import logging
import functools
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
import statistics
from dataclasses import dataclass, field
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("performance")

@dataclass
class PerformanceMetric:
    """
    性能指标数据类
    """
    function_name: str  # 函数名称
    execution_time: float  # 执行时间（秒）
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
class PerformanceMonitor:
    """
    性能监控器类，负责收集和统计性能数据
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化性能监控器
        """
        if not self._initialized:
            self._metrics: List[PerformanceMetric] = []  # 性能指标列表
            self._lock = threading.Lock()  # 线程锁，保证线程安全
            self._stats_cache: Dict[str, Dict[str, Any]] = {}  # 统计缓存
            self._cache_expiry = time.time()  # 缓存过期时间
            self._initialized = True
            logger.info("性能监控器初始化完成")
    
    def record(self, metric: PerformanceMetric) -> None:
        """
        记录性能指标
        
        参数:
            metric: 性能指标对象
        """
        with self._lock:
            self._metrics.append(metric)
            # 当记录新指标时，使缓存失效
            self._cache_expiry = time.time()
            
            # 记录详细日志
            logger.info(
                f"性能指标: {metric.function_name} 执行耗时={metric.execution_time:.4f}秒 "
                f"元数据={', '.join([f'{k}={v}' for k, v in metric.metadata.items()])}"
            )
    
    def get_statistics(self, function_name: Optional[str] = None, 
                     time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        获取性能统计
        
        参数:
            function_name: 如果提供，仅返回指定函数的统计
            time_window: 如果提供，仅统计指定时间窗口（秒）内的指标
            
        返回:
            统计信息字典
        """
        # 检查缓存是否有效
        cache_key = f"{function_name}_{time_window}"
        current_time = time.time()
        if current_time - self._cache_expiry < 60 and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        with self._lock:
            # 筛选指标
            filtered_metrics = self._metrics
            
            if function_name:
                filtered_metrics = [m for m in filtered_metrics if m.function_name == function_name]
            
            if time_window:
                cutoff_time = datetime.now().timestamp() - time_window
                filtered_metrics = [
                    m for m in filtered_metrics 
                    if m.timestamp.timestamp() > cutoff_time
                ]
            
            # 如果没有指标，返回空统计
            if not filtered_metrics:
                stats = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "median_time": 0,
                    "p95_time": 0,
                    "p99_time": 0
                }
                self._stats_cache[cache_key] = stats
                return stats
            
            # 计算统计信息
            execution_times = [m.execution_time for m in filtered_metrics]
            
            stats = {
                "count": len(filtered_metrics),
                "total_time": sum(execution_times),
                "avg_time": statistics.mean(execution_times),
                "min_time": min(execution_times),
                "max_time": max(execution_times),
                "median_time": statistics.median(execution_times)
            }
            
            # 计算百分位数
            if len(execution_times) >= 20:  # 只有样本足够多时才计算
                execution_times.sort()
                stats["p95_time"] = execution_times[int(0.95 * len(execution_times))]
                stats["p99_time"] = execution_times[int(0.99 * len(execution_times))]
            else:
                stats["p95_time"] = stats["max_time"]
                stats["p99_time"] = stats["max_time"]
            
            # 更新缓存
            self._stats_cache[cache_key] = stats
            
            return stats
    
    def get_function_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取按函数分组的统计信息
        
        返回:
            每个函数的统计信息字典
        """
        with self._lock:
            # 获取所有不同的函数名
            function_names = set(m.function_name for m in self._metrics)
            
            # 为每个函数计算统计信息
            result = {}
            for name in function_names:
                result[name] = self.get_statistics(function_name=name)
            
            return result
    
    def clear_old_metrics(self, max_age_hours: int = 24) -> int:
        """
        清除旧的性能指标
        
        参数:
            max_age_hours: 保留的最大小时数
            
        返回:
            清除的指标数量
        """
        with self._lock:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            original_count = len(self._metrics)
            
            self._metrics = [
                m for m in self._metrics 
                if m.timestamp.timestamp() > cutoff_time
            ]
            
            # 清除后使缓存失效
            self._cache_expiry = 0
            self._stats_cache.clear()
            
            return original_count - len(self._metrics)
    
    def print_summary(self, time_window: Optional[int] = None) -> None:
        """
        打印性能统计摘要
        
        参数:
            time_window: 时间窗口（秒）
        """
        function_stats = self.get_function_statistics()
        
        if not function_stats:
            logger.info("没有可用的性能统计数据")
            return
        
        # 打印每个函数的统计信息
        time_window_str = f"过去 {time_window} 秒" if time_window else "所有时间"
        logger.info(f"===== 性能统计摘要 ({time_window_str}) =====")
        
        for func_name, stats in function_stats.items():
            logger.info(
                f"函数: {func_name} - "
                f"调用次数: {stats['count']}, "
                f"平均耗时: {stats['avg_time']:.4f}秒, "
                f"最小: {stats['min_time']:.4f}秒, "
                f"最大: {stats['max_time']:.4f}秒, "
                f"中位数: {stats['median_time']:.4f}秒, "
                f"P95: {stats.get('p95_time', 0):.4f}秒, "
                f"P99: {stats.get('p99_time', 0):.4f}秒"
            )
        
        logger.info("==============================")

def measure_performance(metadata_func: Optional[Callable] = None):
    """
    性能测量装饰器
    
    参数:
        metadata_func: 可选的函数，返回要添加到性能指标的元数据
    
    返回:
        装饰后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 获取元数据
            metadata = {}
            if metadata_func:
                try:
                    metadata = metadata_func(*args, **kwargs) or {}
                except Exception as e:
                    logger.warning(f"获取性能元数据时出错: {e}")
            
            # 测量性能
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录性能指标
                metric = PerformanceMetric(
                    function_name=func.__qualname__,
                    execution_time=execution_time,
                    metadata=metadata
                )
                performance_monitor.record(metric)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 获取元数据
            metadata = {}
            if metadata_func:
                try:
                    metadata = metadata_func(*args, **kwargs) or {}
                except Exception as e:
                    logger.warning(f"获取性能元数据时出错: {e}")
            
            # 测量性能
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录性能指标
                metric = PerformanceMetric(
                    function_name=func.__qualname__,
                    execution_time=execution_time,
                    metadata=metadata
                )
                performance_monitor.record(metric)
        
        # 根据函数类型返回不同的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    # 允许直接使用 @measure_performance 或 @measure_performance()
    if callable(metadata_func) and not isinstance(metadata_func, Callable):
        func = metadata_func
        metadata_func = None
        return decorator(func)
    
    return decorator

# 创建全局性能监控器实例
performance_monitor = PerformanceMonitor() 