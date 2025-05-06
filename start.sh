#!/bin/bash
# SenseVoice API 启动脚本
# 自动配置环境变量并启动服务

# 当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 显示彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印标题
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}     SenseVoice API 启动脚本     ${NC}"
echo -e "${GREEN}================================${NC}"

# 检查CUDA可用性
echo -e "${BLUE}[1/5]${NC} 检查CUDA环境..."
if [ -x "$(command -v nvidia-smi)" ]; then
    echo -e "${GREEN}✓ 找到NVIDIA驱动${NC}"
    nvidia-smi
    
    # 获取可用GPU列表
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader)
    echo -e "${BLUE}可用GPU:${NC}"
    echo "$AVAILABLE_GPUS"
    
    # 取空闲内存最大的GPU
    GPU_ID=$(echo "$AVAILABLE_GPUS" | sort -t ',' -k3 -nr | head -n1 | cut -d',' -f1 | tr -d ' ')
    GPU_NAME=$(echo "$AVAILABLE_GPUS" | sort -t ',' -k3 -nr | head -n1 | cut -d',' -f2)
    GPU_MEM=$(echo "$AVAILABLE_GPUS" | sort -t ',' -k3 -nr | head -n1 | cut -d',' -f3)
    
    echo -e "${GREEN}✓ 自动选择GPU: ${GPU_ID} (${GPU_NAME}, 空闲内存: ${GPU_MEM})${NC}"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export SENSEVOICE_GPU_DEVICE=${GPU_ID}
else
    echo -e "${YELLOW}⚠ 未检测到NVIDIA驱动，将使用CPU模式${NC}"
    export CUDA_VISIBLE_DEVICES=""
    export SENSEVOICE_GPU_DEVICE="-1"
fi

# 设置服务配置
echo -e "${BLUE}[2/5]${NC} 设置服务配置..."

# 检查是否指定了端口
if [ -z "$SENSEVOICE_PORT" ]; then
    export SENSEVOICE_PORT=8000
    echo -e "${YELLOW}ℹ 未指定端口，将使用默认端口: ${SENSEVOICE_PORT}${NC}"
else
    echo -e "${GREEN}✓ 使用指定端口: ${SENSEVOICE_PORT}${NC}"
fi

# 检查是否指定了主机
if [ -z "$SENSEVOICE_HOST" ]; then
    export SENSEVOICE_HOST="0.0.0.0"
    echo -e "${YELLOW}ℹ 未指定主机地址，将使用默认地址: ${SENSEVOICE_HOST}${NC}"
else
    echo -e "${GREEN}✓ 使用指定主机地址: ${SENSEVOICE_HOST}${NC}"
fi

# 设置模型目录
if [ -z "$SENSEVOICE_MODEL_DIR" ]; then
    export SENSEVOICE_MODEL_DIR="iic/SenseVoiceSmall"
    echo -e "${YELLOW}ℹ 未指定模型目录，将使用默认模型: ${SENSEVOICE_MODEL_DIR}${NC}"
else
    echo -e "${GREEN}✓ 使用指定模型目录: ${SENSEVOICE_MODEL_DIR}${NC}"
fi

# 设置日志级别
if [ -z "$SENSEVOICE_LOG_LEVEL" ]; then
    export SENSEVOICE_LOG_LEVEL="INFO"
    echo -e "${YELLOW}ℹ 未指定日志级别，将使用默认级别: ${SENSEVOICE_LOG_LEVEL}${NC}"
else
    echo -e "${GREEN}✓ 使用指定日志级别: ${SENSEVOICE_LOG_LEVEL}${NC}"
fi

# 检查缓存目录
echo -e "${BLUE}[3/5]${NC} 检查临时目录..."
if [ -z "$SENSEVOICE_TEMP_DIR" ]; then
    export SENSEVOICE_TEMP_DIR="/tmp/sensevoice"
    echo -e "${YELLOW}ℹ 未指定临时目录，将使用默认目录: ${SENSEVOICE_TEMP_DIR}${NC}"
else
    echo -e "${GREEN}✓ 使用指定临时目录: ${SENSEVOICE_TEMP_DIR}${NC}"
fi

# 创建临时目录
mkdir -p $SENSEVOICE_TEMP_DIR
echo -e "${GREEN}✓ 临时目录已准备${NC}"

# 检查Python环境
echo -e "${BLUE}[4/5]${NC} 检查Python环境..."
PYTHON_CMD=""

# 优先检查python3是否存在
if [ -x "$(command -v python3)" ]; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ 使用python3${NC}"
else
    # 检查python是否存在且为python3
    if [ -x "$(command -v python)" ]; then
        PY_VERSION=$(python --version 2>&1)
        if [[ $PY_VERSION == *"Python 3"* ]]; then
            PYTHON_CMD="python"
            echo -e "${GREEN}✓ 使用python (${PY_VERSION})${NC}"
        else
            echo -e "${RED}✗ 未找到Python 3，请先安装${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ 未找到Python，请先安装${NC}"
        exit 1
    fi
fi

# 检查依赖项
echo -e "${BLUE}[5/5]${NC} 检查依赖项..."
REQUIRED_PACKAGES=("fastapi" "uvicorn" "numpy" "onnxruntime" "websockets")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON_CMD -c "import $pkg" &> /dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ 缺少以下依赖项:${NC} ${MISSING_PACKAGES[*]}"
    read -p "是否自动安装缺少的依赖项? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}正在安装依赖...${NC}"
        $PYTHON_CMD -m pip install ${MISSING_PACKAGES[@]}
        echo -e "${GREEN}✓ 依赖项安装完成${NC}"
    else
        echo -e "${YELLOW}请手动安装缺少的依赖项后再运行该脚本${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 所有依赖项已安装${NC}"
fi

# 启动API服务
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}      启动 SenseVoice API       ${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${BLUE}主机: ${SENSEVOICE_HOST}${NC}"
echo -e "${BLUE}端口: ${SENSEVOICE_PORT}${NC}"
echo -e "${BLUE}GPU ID: ${SENSEVOICE_GPU_DEVICE}${NC}"
echo -e "${BLUE}模型: ${SENSEVOICE_MODEL_DIR}${NC}"
echo -e "${GREEN}================================${NC}"

# 检查是否以守护模式启动
if [ "$1" == "-d" ] || [ "$1" == "--daemon" ]; then
    echo -e "${YELLOW}以守护模式启动服务...${NC}"
    nohup $PYTHON_CMD main.py > sensevoice_api.log 2>&1 &
    echo $! > sensevoice_api.pid
    echo -e "${GREEN}✓ 服务已启动，PID: $(cat sensevoice_api.pid)${NC}"
    echo -e "${BLUE}可以通过以下命令查看日志:${NC}"
    echo -e "${BLUE}  tail -f sensevoice_api.log${NC}"
else
    echo -e "${YELLOW}以前台模式启动服务...${NC}"
    $PYTHON_CMD main.py 