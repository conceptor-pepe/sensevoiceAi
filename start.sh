#!/bin/bash

# 设置默认的环境变量
export SENSEVOICE_MODEL_DIR="iic/SenseVoiceSmall"
export SENSEVOICE_DEVICE="cuda:0"
export SENSEVOICE_WORKERS=4
export SENSEVOICE_TIMEOUT=30
export SENSEVOICE_MAX_QUEUE=100

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            export SENSEVOICE_MODEL_DIR="$2"
            shift 2
            ;;
        --device)
            export SENSEVOICE_DEVICE="$2"
            shift 2
            ;;
        --workers)
            export SENSEVOICE_WORKERS="$2"
            shift 2
            ;;
        --timeout)
            export SENSEVOICE_TIMEOUT="$2"
            shift 2
            ;;
        --max-queue)
            export SENSEVOICE_MAX_QUEUE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --model-dir PATH     设置SenseVoice模型目录 (默认: iic/SenseVoiceSmall)"
            echo "  --device DEVICE      设置使用的设备 (默认: cuda:0)"
            echo "  --workers NUM        设置工作器数量 (默认: 4)"
            echo "  --timeout SECONDS    设置任务超时时间 (默认: 30)"
            echo "  --max-queue SIZE     设置最大队列长度 (默认: 100)"
            echo "  --port PORT          设置API服务端口 (默认: 8000)"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 如果未设置端口，使用默认端口
if [ -z "$PORT" ]; then
    PORT=8000
fi

# 检查模型目录
if [ ! -d "$SENSEVOICE_MODEL_DIR" ]; then
    echo "警告: 模型目录 $SENSEVOICE_MODEL_DIR 不存在，请确保模型路径正确"
fi

# 检查CUDA可用性
if [[ "$SENSEVOICE_DEVICE" == cuda* ]]; then
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    if [ $? -ne 0 ]; then
        echo "警告: 无法检查CUDA可用性，请确保PyTorch正确安装"
    fi
fi

# 检查依赖项
echo "检查依赖项..."
pip install -r requirements.txt

# 显示配置信息
echo "===================== 配置信息 ====================="
echo "模型目录:       $SENSEVOICE_MODEL_DIR"
echo "设备:           $SENSEVOICE_DEVICE"
echo "工作器数量:     $SENSEVOICE_WORKERS"
echo "任务超时:       $SENSEVOICE_TIMEOUT 秒"
echo "最大队列长度:   $SENSEVOICE_MAX_QUEUE"
echo "API 服务端口:   $PORT"
echo "===================================================="

# 启动 API 服务
echo "正在启动 SenseVoice API 服务..."
uvicorn api:app --host 0.0.0.0 --port $PORT

