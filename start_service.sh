#!/bin/bash
# SenseVoice API服务启动脚本（优化版）
# 注: 此版本已优化性能，去除缓存和临时IO操作，并使用单例模式

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 默认配置
PORT=8000
HOST="0.0.0.0"
DEVICE="cuda:5"
GPU_ID=5  # 直接使用数字形式的GPU ID
WORKERS=1
DEBUG=false
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
      # 从设备字符串中提取GPU ID
      if [[ $DEVICE == cuda:* ]]; then
        GPU_ID="${DEVICE#cuda:}"
      else
        GPU_ID="$DEVICE"
      fi
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
    --debug)
      DEBUG=true
      shift
      ;;
    --cpu)
      DEVICE="cpu"
      GPU_ID="-1"  # 使用-1表示CPU模式
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
      echo "  --debug          启用调试模式"
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
# 设置CUDA可见设备
if [[ $DEVICE == cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "已设置CUDA_VISIBLE_DEVICES=$GPU_ID"
fi

# 设置调试环境变量
if [ "$DEBUG" = true ]; then
  export DEBUG=true
fi

# 创建/确保日志目录存在
LOG_DIR="/var/log/sensevoice"
if [ ! -d "$LOG_DIR" ]; then
  echo "日志目录 $LOG_DIR 不存在，尝试创建..."
  if [ "$(id -u)" -eq 0 ]; then
    # 以root运行
    mkdir -p "$LOG_DIR"
    chmod 755 "$LOG_DIR"
    echo "成功创建日志目录: $LOG_DIR"
  else
    # 尝试使用sudo创建
    echo "需要sudo权限创建日志目录"
    sudo mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
      echo "警告: 无法创建日志目录 $LOG_DIR，将使用标准输出/错误代替"
      export USE_STDERR=true
    else
      sudo chmod 755 "$LOG_DIR"
      current_user=$(whoami)
      sudo chown "$current_user" "$LOG_DIR"
      echo "成功创建日志目录: $LOG_DIR"
    fi
  fi
else
  echo "日志目录已存在: $LOG_DIR"
  # 检查写入权限
  if [ ! -w "$LOG_DIR" ]; then
    echo "警告: 没有 $LOG_DIR 的写入权限"
    if [ "$(id -u)" -eq 0 ]; then
      chmod 755 "$LOG_DIR"
    else
      echo "尝试获取写入权限..."
      sudo chmod 755 "$LOG_DIR"
      if [ $? -ne 0 ]; then
        echo "警告: 无法获取日志目录写入权限，将使用标准输出/错误代替"
        export USE_STDERR=true
      fi
    fi
  fi
fi

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
echo "  - 设备: $DEVICE (GPU ID: $GPU_ID)"
echo "  - 工作进程: $WORKERS"
echo "  - 性能优化: $OPT_LEVEL"
echo "  - 调试模式: $DEBUG"
echo "  - 日志目录: $LOG_DIR"
echo "=========================="

# 检查CUDA环境
if [[ $DEVICE == cuda:* ]]; then
  if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/  /'
  else
    echo "警告: 未检测到NVIDIA GPU工具，但GPU模式已选择"
  fi
fi

# 启动服务
DEBUG_FLAG=""
if [ "$DEBUG" = true ]; then
  DEBUG_FLAG="--debug"
fi

# 使用直接的GPU ID参数，不再使用字符串替换
python main.py --host "$HOST" --port "$PORT" --workers "$WORKERS" --gpu "$GPU_ID" $DEBUG_FLAG 