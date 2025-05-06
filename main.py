#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 应用入口
启动API服务
"""

import os
import sys
import uvicorn
import config
from logger import logger

# 显式配置GPU环境
def configure_gpu():
    """配置GPU环境"""
    # 记录环境变量
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
    sensevoice_gpu = os.environ.get("SENSEVOICE_GPU_DEVICE", "未设置")
    logger.info(f"启动时环境变量: CUDA_VISIBLE_DEVICES={gpu_device}, SENSEVOICE_GPU_DEVICE={sensevoice_gpu}")
    
    # 确保设置了CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_DEVICE
        logger.info(f"未设置CUDA_VISIBLE_DEVICES，已设置为 {config.GPU_DEVICE}")
    
    # 设置ONNX Runtime环境
    try:
        import onnxruntime as ort
        
        # 获取可用的提供程序列表
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime可用提供程序: {providers}")
        
        # 如果支持CUDA，配置CUDA提供程序
        if 'CUDAExecutionProvider' in providers:
            # 获取和打印GPU信息
            gpu_info = getattr(ort, 'get_device', lambda: '未知设备信息')()
            logger.info(f"ONNX Runtime设备信息: {gpu_info}")
            
            # 确认期望使用的GPU设备
            target_gpu = int(config.GPU_DEVICE)
            logger.info(f"目标使用GPU设备ID: {target_gpu}")
            
            # 设置提供程序选项
            provider_options = {
                'device_id': target_gpu,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            
            # 1. 首先设置CUDA_VISIBLE_DEVICES环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
            logger.info(f"已设置CUDA_VISIBLE_DEVICES={target_gpu}")
            
            # 2. 设置ONNX Runtime特定环境变量
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # 启用FP16
            os.environ["ORT_CUDA_DEVICE"] = str(target_gpu)  # 设置CUDA设备ID
            
            # 3. 额外设置供onnxruntime使用的环境变量
            os.environ["OMP_NUM_THREADS"] = "1"  # 限制OMP线程数
            os.environ["ORT_THREAD_POOL_ALLOW_SPINNING"] = "0"  # 禁用线程池自旋
            
            logger.info(f"已配置ONNX Runtime CUDA提供程序选项: {provider_options}")
            logger.info(f"已设置环境变量: ORT_TENSORRT_FP16_ENABLE=1, ORT_CUDA_DEVICE={target_gpu}")
            
            # 尝试设置ONNX Runtime日志级别，不再调用不存在的set_session_options方法
            try:
                if hasattr(ort, 'set_default_logger_severity'):
                    ort.set_default_logger_severity(0)  # 设置日志级别为详细
                    logger.info("已设置ONNX Runtime日志级别为详细")
            except Exception as e:
                logger.warning(f"无法设置ONNX Runtime日志级别: {str(e)}")
        else:
            logger.warning("ONNX Runtime不支持CUDA，将使用CPU推理")
    except ImportError as e:
        logger.error(f"无法导入onnxruntime: {str(e)}")
    except Exception as e:
        logger.error(f"配置GPU时出错: {str(e)}")

def main():
    """
    应用主入口
    """
    # 配置GPU环境
    configure_gpu()
    
    # 打印服务配置信息
    config.print_config()
    
    # 导入应用（在配置完GPU后导入，确保正确配置）
    from api import app
    
    # 启动服务
    logger.info(f"启动服务: host={config.API_HOST}, port={config.API_PORT}")
    uvicorn.run(
        app, 
        host=config.API_HOST, 
        port=config.API_PORT
    )

if __name__ == "__main__":
    main() 