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

# 显示配置信息
echo "启动SenseVoice API服务..."
echo "模型目录: $MODEL_DIR"
echo "GPU设备: $GPU_DEVICE"
echo "主机地址: $HOST"
echo "端口: $PORT"
echo "批处理大小: $BATCH_SIZE"

# 设置环境变量
export SENSEVOICE_MODEL_DIR=$MODEL_DIR
export SENSEVOICE_GPU_DEVICE=$GPU_DEVICE
export SENSEVOICE_HOST=$HOST
export SENSEVOICE_PORT=$PORT
export SENSEVOICE_BATCH_SIZE=$BATCH_SIZE
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# 启动服务
python api.py 