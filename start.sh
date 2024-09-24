#!/bin/bash
# SenseVoice ASR API 开发调试启动脚本

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 日志函数
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 默认参数
HOST="0.0.0.0"
PORT="8000"
GPU_ID="5"
DEBUG="true"
CACHE="true"

# 显示用法
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --host HOST      设置监听主机 (默认: $HOST)"
    echo "  -p, --port PORT      设置监听端口 (默认: $PORT)"
    echo "  -g, --gpu GPU_ID     设置GPU ID (默认: $GPU_ID)"
    echo "  -d, --debug          启用调试模式 (默认: 启用)"
    echo "  -nc, --no-cache      禁用缓存 (默认: 启用)"
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
        -nd|--no-debug)
            DEBUG="false"
            shift
            ;;
        -nc|--no-cache)
            CACHE="false"
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
if ! command -v nvidia-smi &> /dev/null; then
    warning "未检测到NVIDIA GPU工具，模型可能无法正常运行"
fi

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    warning "未找到requirements.txt文件，可能需要先安装依赖"
else
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        log "提示: 建议使用虚拟环境运行 (python -m venv venv && source venv/bin/activate)"
    fi
fi

# 创建日志目录
mkdir -p logs

# 构建启动命令
DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
    log "已启用调试模式"
fi

CACHE_FLAG="--cache $CACHE"
if [ "$CACHE" = "false" ]; then
    log "已禁用缓存"
fi

# 显示启动信息
log "正在启动SenseVoice ASR API服务..."
log "主机: $HOST"
log "端口: $PORT"
log "GPU ID: $GPU_ID"

# 启动服务
log "执行命令: python3 main.py --host $HOST --port $PORT --gpu $GPU_ID $CACHE_FLAG $DEBUG_FLAG"
python3 main.py --host "$HOST" --port "$PORT" --gpu "$GPU_ID" $CACHE_FLAG $DEBUG_FLAG

# 脚本结束
exit $? 