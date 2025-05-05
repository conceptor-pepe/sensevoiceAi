#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 获取安装路径
INSTALL_PATH=$(pwd)
SERVICE_FILE="$INSTALL_PATH/sensevoice.service"
SYSTEMD_PATH="/etc/systemd/system/sensevoice.service"

# 检查是否为root用户
if [ "$(id -u)" != "0" ]; then
    echo -e "${RED}错误: 此脚本必须以root权限运行${NC}"
    echo "请使用 'sudo $0' 重新运行"
    exit 1
fi

# 显示欢迎信息
echo -e "${GREEN}=== SenseVoice API 服务安装工具 ===${NC}"
echo ""

# 检查必要文件
echo -e "${YELLOW}检查必要文件...${NC}"
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}错误: 找不到服务文件: $SERVICE_FILE${NC}"
    exit 1
fi

if [ ! -f "$INSTALL_PATH/main.py" ]; then
    echo -e "${RED}错误: 找不到主程序文件: $INSTALL_PATH/main.py${NC}"
    exit 1
fi

# 创建日志目录
echo -e "${YELLOW}创建必要目录...${NC}"
mkdir -p /var/log/sensevoice
mkdir -p /var/run/sensevoice
echo "已创建日志目录: /var/log/sensevoice"
echo "已创建运行目录: /var/run/sensevoice"

# 更新服务文件中的路径
echo -e "${YELLOW}更新服务配置...${NC}"
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_PATH|g" "$SERVICE_FILE"
sed -i "s|Documentation=file:///.*|Documentation=file://$INSTALL_PATH/README.md|g" "$SERVICE_FILE"

# 拷贝服务文件到系统目录
echo -e "${YELLOW}安装服务文件...${NC}"
cp "$SERVICE_FILE" "$SYSTEMD_PATH"
echo "已安装服务文件: $SYSTEMD_PATH"

# 创建环境变量配置文件目录（可选）
if [ ! -d "/etc/sensevoice" ]; then
    mkdir -p /etc/sensevoice
    echo "已创建配置目录: /etc/sensevoice"
    
    # 创建示例环境变量文件
    cat > /etc/sensevoice/env.conf.example << EOF
# SenseVoice API 环境变量配置示例
# 将此文件复制为 env.conf 并取消注释需要的选项进行自定义

# SENSEVOICE_MODEL_DIR=iic/SenseVoiceSmall
# SENSEVOICE_GPU_DEVICE=0
# SENSEVOICE_HOST=0.0.0.0
# SENSEVOICE_PORT=8000
# SENSEVOICE_BATCH_SIZE=1
# SENSEVOICE_LOG_LEVEL=INFO
# SENSEVOICE_TEMP_DIR=/var/log/sensevoice
# CUDA_VISIBLE_DEVICES=0
EOF
    echo "已创建环境变量示例文件: /etc/sensevoice/env.conf.example"
fi

# 重新加载systemd
echo -e "${YELLOW}重新加载systemd...${NC}"
systemctl daemon-reload
echo "systemd已重新加载"

# 启用服务（开机自启）
echo -e "${YELLOW}启用服务...${NC}"
systemctl enable sensevoice.service
echo "服务已启用，将在系统启动时自动启动"

# 询问是否立即启动服务
echo ""
read -p "是否现在启动服务? (y/n): " choice
if [[ "$choice" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}启动服务...${NC}"
    systemctl start sensevoice.service
    sleep 3
    
    # 检查服务状态
    if systemctl is-active --quiet sensevoice.service; then
        echo -e "${GREEN}服务已成功启动!${NC}"
        systemctl status sensevoice.service
    else
        echo -e "${RED}服务启动失败。请检查日志: journalctl -u sensevoice.service${NC}"
        systemctl status sensevoice.service
    fi
fi

# 显示服务使用说明
echo ""
echo -e "${GREEN}=== 服务管理命令 ===${NC}"
echo "启动服务:   sudo systemctl start sensevoice.service"
echo "停止服务:   sudo systemctl stop sensevoice.service"
echo "重启服务:   sudo systemctl restart sensevoice.service"
echo "查看状态:   sudo systemctl status sensevoice.service"
echo "查看日志:   sudo journalctl -u sensevoice.service"
echo "启用服务:   sudo systemctl enable sensevoice.service"
echo "禁用服务:   sudo systemctl disable sensevoice.service"
echo ""
echo -e "${GREEN}安装完成!${NC}" 