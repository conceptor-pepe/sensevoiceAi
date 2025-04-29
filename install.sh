#!/bin/bash
# install.sh - SenseVoice API 安装脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${GREEN}SenseVoice API 安装脚本${NC}"
echo "============================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "检测到Python版本: ${YELLOW}$python_version${NC}"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA可用${NC}"
    nvidia-smi | head -n 3
else
    echo -e "${YELLOW}警告: 未检测到NVIDIA GPU，将使用CPU模式${NC}"
fi

# 检查是否存在名为 'sensevoice' 的 Conda 环境
if conda env list | grep -q '^sensevoice\s'; then
    echo -e "\n${YELLOW}检测到 Conda 环境: sensevoice${NC}"
else
    # 创建环境
    echo -e "\n${GREEN}创建 Conda 环境 sensevoice${NC}"
    conda create -n sensevoice python=3.10 -y
fi

# 激活环境（无论是否已存在）
source $(conda info --base)/etc/profile.d/conda.sh  # 确保 conda activate 可用
conda activate sensevoice
echo -e "当前 Python 路径: ${YELLOW}$(which python)${NC}"


# 安装依赖
echo -e "\n${GREEN}安装依赖${NC}"
pip install -r requirements.txt

# 下载模型
echo -e "\n${GREEN}初始化SenseVoice模型${NC}"
python init_model.py

# 测试安装
echo -e "\n${GREEN}测试安装${NC}"
python -c "from funasr_onnx import SenseVoiceSmall; print('SenseVoice导入成功')"

echo -e "\n${GREEN}安装完成${NC}"
echo "启动服务: ./start.sh"
echo "停止服务: ./stop.sh"
echo "查看状态: ./status.sh"