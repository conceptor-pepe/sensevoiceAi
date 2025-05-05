#!/bin/bash

# 设置默认参数
MODEL_DIR=${SENSEVOICE_MODEL_DIR:-"iic/SenseVoiceSmall"}

# 设置GPU设备
GPU_DEVICE=${SENSEVOICE_GPU_DEVICE:-"5"}

# 设置主机地址
HOST=${SENSEVOICE_HOST:-"0.0.0.0"}

# 设置端口
PORT=${SENSEVOICE_PORT:-"8000"}

# 设置批处理大小
BATCH_SIZE=${SENSEVOICE_BATCH_SIZE:-"1"}

# 设置日志级别
LOG_LEVEL=${SENSEVOICE_LOG_LEVEL:-"INFO"}

# 设置日志格式
LOG_FORMAT=${SENSEVOICE_LOG_FORMAT:-"%(asctime)s - %(name)s - %(levelname)s - %(message)s"}

# 设置日志文件（默认不输出到文件）
LOG_FILE=${SENSEVOICE_LOG_FILE:-""}

# 设置临时目录
TEMP_DIR=${SENSEVOICE_TEMP_DIR:-"/var/log/sensevoice"}

# 显示配置信息
echo "启动SenseVoice API服务..."
echo "模型目录: $MODEL_DIR"
echo "GPU设备: $GPU_DEVICE"
echo "主机地址: $HOST"
echo "端口: $PORT"
echo "批处理大小: $BATCH_SIZE"
echo "日志级别: $LOG_LEVEL"
if [ -n "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
    # 确保日志目录存在
    LOG_DIR=$(dirname "$LOG_FILE")
    mkdir -p "$LOG_DIR"
fi

# 设置环境变量
export SENSEVOICE_MODEL_DIR=$MODEL_DIR
export SENSEVOICE_GPU_DEVICE=$GPU_DEVICE
export SENSEVOICE_HOST=$HOST
export SENSEVOICE_PORT=$PORT
export SENSEVOICE_BATCH_SIZE=$BATCH_SIZE
export SENSEVOICE_LOG_LEVEL=$LOG_LEVEL
export SENSEVOICE_LOG_FORMAT="$LOG_FORMAT"
export SENSEVOICE_LOG_FILE="$LOG_FILE"
export SENSEVOICE_TEMP_DIR="$TEMP_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# 记录启动时间戳
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "服务启动时间: $START_TIME"

# 启动服务
python main.py 