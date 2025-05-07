#!/bin/bash
# SenseVoice API服务启动脚本（优化版）
# 注: 此版本已优化性能，去除缓存和临时IO操作，并使用单例模式

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 默认配置
PORT=8000
HOST="0.0.0.0"
DEVICE="cuda:5"
WORKERS=1
# 性能优化选项
OPT_LEVEL="high" # 性能优化级别: low, medium, high

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    --workers=*)
      WORKERS="${1#*=}"
      shift
      ;;
    --opt=*)
      OPT_LEVEL="${1#*=}"
      shift
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    --help)
      echo "SenseVoice API服务启动脚本（优化版）"
      echo ""
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --port=PORT      指定端口号 (默认: 8000)"
      echo "  --host=HOST      指定监听地址 (默认: 0.0.0.0)"
      echo "  --device=DEVICE  指定运行设备 (默认: cuda:5)"
      echo "  --workers=N      指定工作进程数 (默认: 1)"
      echo "  --opt=LEVEL      指定性能优化级别: low, medium, high (默认: high)"
      echo "  --cpu            使用CPU运行 (等同于 --device=cpu)"
      echo "  --help           显示帮助信息"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

# 设置设备环境变量
export SENSEVOICE_DEVICE=$DEVICE

# 根据优化级别设置环境变量
case $OPT_LEVEL in
  "low")
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    ;;
  "medium")
    export OMP_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    ;;
  "high")
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export ORT_TENSORRT_FP16_ENABLE=1
    ;;
  *)
    echo "无效的优化级别: $OPT_LEVEL，使用默认级别(high)"
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export ORT_TENSORRT_FP16_ENABLE=1
    ;;
esac

echo "=== SenseVoice API服务（优化版）==="
echo "启动配置:"
echo "  - 监听地址: $HOST"
echo "  - 端口: $PORT"
echo "  - 设备: $DEVICE"
echo "  - 工作进程: $WORKERS"
echo "  - 性能优化: $OPT_LEVEL"
echo "=========================="

# 启动服务
python api.py --host $HOST --port $PORT --workers $WORKERS 