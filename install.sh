#!/bin/bash
# SenseVoice ASR API安装脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 安装路径
INSTALL_DIR="/opt/senseaudio"
SERVICE_NAME="senseaudio"
SERVICE_FILE="${SERVICE_NAME}.service"

# 日志函数
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    error "请使用root权限运行此脚本"
fi

# 显示欢迎信息
log "欢迎使用SenseVoice ASR API安装程序"
log "此脚本将安装语音识别服务到 ${INSTALL_DIR}"
echo ""

# 检查GPU环境
log "检查CUDA环境..."
if ! command -v nvidia-smi &> /dev/null; then
    error "未检测到NVIDIA GPU工具，请确保安装了NVIDIA驱动和CUDA"
fi

PYTHON_VERSION=$(python3 --version 2>&1)
log "检测到Python版本: ${PYTHON_VERSION}"

# 创建安装目录
log "创建安装目录..."
mkdir -p ${INSTALL_DIR}
mkdir -p ${INSTALL_DIR}/logs

# 复制文件
log "复制应用文件..."
cp -f *.py ${INSTALL_DIR}/
cp -f requirements.txt ${INSTALL_DIR}/
cp -f ${SERVICE_FILE} ${INSTALL_DIR}/

# 安装依赖
log "安装Python依赖..."
pip3 install -r requirements.txt

# 设置权限
log "设置文件权限..."
chmod +x ${INSTALL_DIR}/main.py
chown -R nobody:nogroup ${INSTALL_DIR}

# 安装systemd服务
log "配置systemd服务..."
cp -f ${SERVICE_FILE} /etc/systemd/system/
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

# 启动服务
log "启动服务..."
systemctl start ${SERVICE_NAME}
sleep 3

# 检查服务状态
if systemctl is-active --quiet ${SERVICE_NAME}; then
    log "服务已成功启动!"
    log "可通过以下命令查看日志:"
    log "  journalctl -u ${SERVICE_NAME} -f"
    log ""
    log "API地址: http://localhost:8000"
    log "API文档: http://localhost:8000/docs"
else
    error "服务启动失败，请检查日志: journalctl -u ${SERVICE_NAME} -e"
fi

log "安装完成!" 