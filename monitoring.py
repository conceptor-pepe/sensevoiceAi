import psutil
import os
import logging
import time
import threading
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import traceback

# 尝试导入GPU监控模块
try:
    import torch
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 配置日志
from logging_config import configure_logging
logger = logging.getLogger("monitoring")

class SystemMonitor:
    """
    系统监控类，收集系统和GPU资源使用信息
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(SystemMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, interval: int = 60):
        """
        初始化系统监控器
        
        参数:
            interval: 监控采集间隔（秒）
        """
        if not self._initialized:
            self.interval = interval
            self.enabled = True
            self._metrics: List[Dict[str, Any]] = []
            self._lock = threading.Lock()
            self._max_metrics = 1440  # 存储24小时的数据（按1分钟一次）
            self._thread = None
            self._log_dir = 'logs'
            
            # 确保日志目录存在
            if not os.path.exists(self._log_dir):
                os.makedirs(self._log_dir)
            
            # 初始化GPU设备列表
            self.gpu_devices = []
            if GPU_AVAILABLE:
                try:
                    self.gpu_devices = GPUtil.getGPUs()
                    logger.info(f"检测到 {len(self.gpu_devices)} 个GPU设备")
                except Exception as e:
                    logger.warning(f"获取GPU设备信息失败: {str(e)}")
            
            self._initialized = True
            logger.info(f"系统监控器初始化完成，采集间隔: {interval}秒")
    
    def start(self) -> bool:
        """
        启动监控线程
        
        返回:
            是否成功启动
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("监控线程已在运行中")
            return False
        
        self.enabled = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("系统监控线程已启动")
        return True
    
    def stop(self) -> bool:
        """
        停止监控线程
        
        返回:
            是否成功停止
        """
        if self._thread is None or not self._thread.is_alive():
            logger.warning("监控线程未运行")
            return False
        
        self.enabled = False
        self._thread.join(timeout=5.0)
        logger.info("系统监控线程已停止")
        return True
    
    def _monitoring_loop(self):
        """
        监控循环，定期采集系统指标
        """
        while self.enabled:
            try:
                # 采集系统指标
                metrics = self.collect_metrics()
                
                with self._lock:
                    self._metrics.append(metrics)
                    # 限制存储的指标数量
                    if len(self._metrics) > self._max_metrics:
                        self._metrics.pop(0)
                
                # 每小时记录一次到文件
                current_hour = datetime.now().hour
                if datetime.now().minute == 0:
                    self._save_metrics_to_file()
                    
            except Exception as e:
                logger.error(f"采集系统指标时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 等待下一个间隔
            time.sleep(self.interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        采集当前系统指标
        
        返回:
            系统指标字典
        """
        timestamp = datetime.now().isoformat()
        
        # 收集CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 收集内存信息
        memory = psutil.virtual_memory()
        
        # 收集磁盘信息
        disk = psutil.disk_usage('/')
        
        # 收集网络信息
        net_io = psutil.net_io_counters()
        
        # 基本系统指标
        metrics = {
            "timestamp": timestamp,
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "freq_current": cpu_freq.current if cpu_freq else None,
                "freq_max": cpu_freq.max if cpu_freq else None
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
        
        # 如果GPU可用，收集GPU信息
        if GPU_AVAILABLE and self.gpu_devices:
            try:
                # 更新GPU设备列表
                self.gpu_devices = GPUtil.getGPUs()
                
                gpu_metrics = []
                for i, gpu in enumerate(self.gpu_devices):
                    gpu_info = {
                        "id": i,
                        "name": gpu.name,
                        "load": gpu.load,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_util": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature
                    }
                    gpu_metrics.append(gpu_info)
                
                metrics["gpu"] = gpu_metrics
                
                # 如果torch可用，收集CUDA内存信息
                if torch.cuda.is_available():
                    cuda_metrics = []
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            allocated = torch.cuda.memory_allocated()
                            reserved = torch.cuda.memory_reserved()
                            max_allocated = torch.cuda.max_memory_allocated()
                            
                            cuda_info = {
                                "device": i,
                                "allocated": allocated,
                                "reserved": reserved,
                                "max_allocated": max_allocated
                            }
                            cuda_metrics.append(cuda_info)
                    
                    metrics["cuda"] = cuda_metrics
                
            except Exception as e:
                logger.error(f"收集GPU指标时出错: {str(e)}")
                logger.error(traceback.format_exc())
        
        # 记录进程信息
        try:
            current_process = psutil.Process(os.getpid())
            process_info = {
                "pid": current_process.pid,
                "cpu_percent": current_process.cpu_percent(interval=None),
                "memory_percent": current_process.memory_percent(),
                "memory_rss": current_process.memory_info().rss,
                "threads": current_process.num_threads(),
                "open_files": len(current_process.open_files())
            }
            metrics["process"] = process_info
        except Exception as e:
            logger.error(f"收集进程指标时出错: {str(e)}")
        
        return metrics
    
    def _save_metrics_to_file(self):
        """
        将指标保存到文件
        """
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            file_path = os.path.join(self._log_dir, f'system_metrics_{date_str}.json')
            
            with self._lock:
                with open(file_path, 'w') as f:
                    json.dump(self._metrics, f, indent=2)
            
            logger.info(f"系统指标已保存到文件: {file_path}")
        except Exception as e:
            logger.error(f"保存系统指标到文件时出错: {str(e)}")
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的系统指标
        
        返回:
            最新的系统指标或None
        """
        with self._lock:
            if not self._metrics:
                return None
            return self._metrics[-1]
    
    def get_metrics_history(self, count: int = 60) -> List[Dict[str, Any]]:
        """
        获取历史系统指标
        
        参数:
            count: 返回的指标数量
            
        返回:
            历史系统指标列表
        """
        with self._lock:
            return self._metrics[-count:] if self._metrics else []
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取系统概况
        
        返回:
            系统概况字典
        """
        latest = self.get_latest_metrics()
        if not latest:
            return {"error": "No metrics available"}
        
        # 简化的系统概况
        summary = {
            "timestamp": latest["timestamp"],
            "cpu_percent": latest["cpu"]["percent"],
            "memory_percent": latest["memory"]["percent"],
            "disk_percent": latest["disk"]["percent"]
        }
        
        # 添加GPU信息（如果有）
        if "gpu" in latest and latest["gpu"]:
            gpu_summary = []
            for gpu in latest["gpu"]:
                gpu_summary.append({
                    "id": gpu["id"],
                    "name": gpu["name"],
                    "load": gpu["load"],
                    "memory_util": gpu["memory_util"],
                    "temperature": gpu["temperature"]
                })
            summary["gpu"] = gpu_summary
        
        return summary


# 创建全局监控器实例
system_monitor = SystemMonitor() 