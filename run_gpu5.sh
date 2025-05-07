#!/bin/bash
# SenseVoice ASR API 5号GPU专用启动脚本

# 设置颜色常量
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

# 固定为5号GPU，不允许修改
GPU_ID=5
HOST="0.0.0.0"
PORT="8000"
DEBUG="false"

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
        -d|--debug)
            DEBUG="true"
            shift
            ;;
        *)
            warning "未知参数: $1"
            shift
            ;;
    esac
done

# 检查环境
log "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    error "未找到python3命令"
fi

# 强制设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export SENSEVOICE_DEVICE="cuda:$GPU_ID" 
export SENSEVOICE_GPU_ID=$GPU_ID

# 设置ONNX优化参数
export ONNX_INTER_OP_THREADS=1
export ONNX_INTRA_OP_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# 强制使用CUDA执行提供者
export ONNXRUNTIME_PROVIDER="CUDAExecutionProvider"
export ORT_TENSORRT_FP16_ENABLE=1
export ORT_TENSORRT_ENABLE=1

# 检查CUDA环境
log "检查CUDA环境和5号GPU状态..."
if nvidia-smi -i $GPU_ID &> /dev/null; then
    log "5号GPU可用"
    nvidia-smi -i $GPU_ID
else
    error "5号GPU不可用，请检查GPU配置"
fi

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
log "GPU: 固定使用5号GPU"

# 启动服务
log "执行命令: python3 main.py --host $HOST --port $PORT --gpu $GPU_ID $DEBUG_FLAG"
python3 main.py --host "$HOST" --port "$PORT" --gpu "$GPU_ID" $DEBUG_FLAG

# 脚本结束
exit $? 