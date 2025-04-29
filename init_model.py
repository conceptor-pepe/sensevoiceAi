#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
init_model.py - SenseVoice模型初始化脚本

该脚本用于下载和初始化SenseVoice语音识别模型。
适用于初次安装或更新SenseVoice API时使用。
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import traceback
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("init_model")

def check_dependencies():
    """
    检查必要的依赖项是否已安装
    
    返回:
        bool: 依赖项检查是否通过
    """
    required_packages = ["funasr_onnx", "funasr_torch", "torch", "torchaudio"]
    missing_packages = []
    
    logger.info("检查依赖项...")
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  √ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"  × {package} 未安装")
    
    if missing_packages:
        logger.warning(f"缺少以下依赖包: {', '.join(missing_packages)}")
        logger.info("尝试安装缺失的依赖...")
        
        for package in missing_packages:
            try:
                logger.info(f"安装 {package}...")
                os.system(f"pip install {package}")
                logger.info(f"{package} 安装完成")
            except Exception as e:
                logger.error(f"安装 {package} 失败: {str(e)}")
                return False
    
    return True

def get_model_dir(args):
    """
    获取模型目录路径
    
    参数:
        args: 命令行参数
        
    返回:
        str: 模型目录路径
    """
    # 优先使用命令行指定的模型目录
    if args.model_dir:
        return args.model_dir
    
    # 其次尝试从config.py读取
    try:
        from config import settings
        return settings.MODEL_DIR
    except (ImportError, AttributeError):
        # 默认使用ModelScope模型
        return "iic/SenseVoiceSmall"

def download_model(model_dir):
    """
    下载并初始化SenseVoice模型
    
    参数:
        model_dir: 模型目录或ModelScope模型ID
        
    返回:
        bool: 模型下载是否成功
    """
    logger.info(f"开始下载/初始化模型: {model_dir}")
    start_time = time.time()
    
    try:
        # 先尝试torch版本
        try:
            logger.info("尝试初始化PyTorch版本模型...")
            from funasr_torch import SenseVoiceSmall
            model = SenseVoiceSmall(model_dir=model_dir)
            logger.info("PyTorch版本模型初始化成功!")
            return True
        except ImportError:
            logger.warning("PyTorch版本不可用，尝试ONNX版本...")
        except Exception as e:
            logger.error(f"PyTorch版本初始化失败: {str(e)}")
        
        # 再尝试onnx版本
        try:
            logger.info("尝试初始化ONNX版本模型...")
            from funasr_onnx import SenseVoiceSmall
            model = SenseVoiceSmall(
                model_dir=model_dir,
                batch_size=1,
                quantize=False,
                providers=["CPUExecutionProvider"]
            )
            logger.info("ONNX版本模型初始化成功!")
            return True
        except ImportError:
            logger.error("ONNX版本不可用，请安装funasr_onnx")
            return False
        except Exception as e:
            logger.error(f"ONNX版本初始化失败: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"模型初始化耗时: {elapsed:.2f}秒")

def test_model(model_dir):
    """
    测试模型加载是否正常
    
    参数:
        model_dir: 模型目录
        
    返回:
        bool: 测试是否通过
    """
    logger.info("测试模型功能...")
    
    # 检查ModelScope缓存目录
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
    actual_model_dir = os.path.join(cache_dir, model_dir) if model_dir.startswith("iic/") else model_dir
    
    if os.path.exists(actual_model_dir):
        logger.info(f"模型目录存在: {actual_model_dir}")
        # 检查目录内容
        files = os.listdir(actual_model_dir)
        logger.info(f"模型目录中的文件: {', '.join(files) if files else '目录为空'}")
    else:
        logger.warning(f"模型目录不存在: {actual_model_dir}")
    
    # 使用真实的音频文件进行测试
    test_audio_file = "en.mp3"
    if not os.path.exists(test_audio_file):
        logger.error(f"测试音频文件不存在: {test_audio_file}")
        return False
    
    logger.info(f"使用音频文件进行测试: {test_audio_file}")
    
    try:
        # 方法1: 使用本地model.py中的SenseVoiceSmall
        logger.info("尝试使用本地model.py中的模型...")
        try:
            import sys
            sys.path.append(os.getcwd())  # 确保能找到当前目录的model.py
            
            from model import SenseVoiceSmall
            
            model = SenseVoiceSmall(model_dir=model_dir)
            
            result = model.infer(test_audio_file, language="auto")
            
            logger.info(f"使用本地模型测试成功，结果: {result}")
            return True
            
        except ImportError as e:
            logger.warning(f"无法导入本地model.py: {str(e)}")
        except Exception as e:
            logger.error(f"使用本地模型测试失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 方法2: 直接使用funasr_torch
        logger.info("尝试直接使用funasr_torch...")
        try:
            from funasr_torch import SenseVoiceSmall
            
            model = SenseVoiceSmall(
                model_dir=model_dir,
                batch_size=1,
                device=os.getenv("SENSEVOICE_DEVICE", "cpu")
            )
            
            result = model.infer(test_audio_file, language="auto")
            
            logger.info(f"使用funasr_torch测试成功，结果: {result}")
            return True
            
        except ImportError:
            logger.warning("无法导入funasr_torch，尝试ONNX版本...")
        except Exception as e:
            logger.error(f"使用funasr_torch失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 方法3: 尝试使用funasr_onnx
        logger.info("尝试使用funasr_onnx...")
        try:
            from funasr_onnx import SenseVoiceSmall
            
            model = SenseVoiceSmall(
                model_dir=model_dir,
                batch_size=1,
                quantize=False,
                providers=["CPUExecutionProvider"]
            )
            
            # 创建音频输入数据
            import numpy as np
            import torchaudio
            
            # 加载音频文件，获取采样率
            waveform, sample_rate = torchaudio.load(test_audio_file)
            waveform = waveform.numpy().mean(axis=0)  # 转为单声道
            feats_len = np.array([len(waveform)])
            
            # ONNX版本的模型需要提供正确的参数
            result = model.infer(
                waveform,  # 音频数据
                feats_len,  # 长度
                language="auto",  # 语言
                textnorm=True  # 文本规范化
            )
            
            logger.info(f"使用funasr_onnx测试成功，结果: {result}")
            return True
            
        except ImportError:
            logger.error("无法导入funasr_onnx")
        except Exception as e:
            logger.error(f"使用funasr_onnx失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.error("所有测试方法均失败")
        return False
        
    except Exception as e:
        logger.error(f"模型测试过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='SenseVoice模型初始化工具')
    parser.add_argument('--model-dir', type=str, default='',
                        help='模型目录或ModelScope模型ID (默认: config.py中的设置)')
    parser.add_argument('--skip-test', action='store_true',
                        help='跳过模型测试')
    parser.add_argument('--force', action='store_true',
                        help='强制重新下载模型')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖项检查失败，请手动安装所需依赖")
        sys.exit(1)
    
    # 获取模型目录
    model_dir = get_model_dir(args)
    logger.info(f"使用模型: {model_dir}")
    
    # 下载/初始化模型
    if not download_model(model_dir):
        logger.error("模型初始化失败")
        sys.exit(1)
    
    # 测试模型
    if not args.skip_test:
        if not test_model(model_dir):
            logger.warning("模型测试失败，但初始化过程已完成")
    
    logger.info("模型初始化完成！")
    logger.info("现在可以通过'./start.sh'启动SenseVoice API服务")

if __name__ == "__main__":
    main() 