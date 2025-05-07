#!/bin/bash
# SenseVoice ASR API GPU启动脚本

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 日志函数
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 默认参数 - 设置为5号GPU
HOST="0.0.0.0"
PORT="8000"
GPU_ID="5"  # 默认使用5号GPU
DEBUG="false"

# 显示用法
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --host HOST      设置监听主机 (默认: $HOST)"
    echo "  -p, --port PORT      设置监听端口 (默认: $PORT)"
    echo "  -g, --gpu GPU_ID     设置GPU ID (默认: $GPU_ID)"
    echo "  -d, --debug          启用调试模式"
    echo "  --help               显示帮助信息"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG="true"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            warning "未知参数: $1"
            usage
            ;;
    esac
done

# 检查环境
log "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    error "未找到python3命令"
fi

# 检查CUDA环境
log "检查CUDA环境..."
if ! nvidia-smi &> /dev/null; then
    warning "未检测到NVIDIA GPU工具，将使用CPU模式"
    GPU_MODE="cpu"
else
    log "NVIDIA GPU工具可用，使用GPU模式"
    # 输出可用GPU信息
    nvidia-smi --list-gpus
    info "可用的GPU设备列表如上"
    GPU_MODE="cuda:$GPU_ID"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export SENSEVOICE_DEVICE=$GPU_MODE
export SENSEVOICE_GPU_ID=$GPU_ID

# 设置ONNX运行时优化参数
export ONNX_INTER_OP_THREADS=1
export ONNX_INTRA_OP_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 强制使用CUDA执行提供者
export ONNXRUNTIME_PROVIDER="CUDAExecutionProvider"
export ORT_TENSORRT_FP16_ENABLE=1
export ORT_TENSORRT_ENABLE=1

# 创建日志目录
mkdir -p logs

# 构建启动命令
DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
    log "已启用调试模式"
fi

# 显示启动信息
log "正在启动SenseVoice ASR API服务..."
log "主机: $HOST"
log "端口: $PORT"
log "设备: $GPU_MODE (GPU ID: $GPU_ID)"

# 启动服务
log "执行命令: python3 main.py --host $HOST --port $PORT --gpu $GPU_ID $DEBUG_FLAG"
python3 main.py --host "$HOST" --port "$PORT" --gpu "$GPU_ID" $DEBUG_FLAG

# 脚本结束
exit $? 