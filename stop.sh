#!/bin/bash
# SenseVoice API 停止脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 服务名称
SERVICE_NAME="SenseVoice API"
PID_FILE="sensevoice_api.pid"

# 显示使用方法
show_usage() {
    echo -e "${YELLOW}使用方法: $0 [参数]${NC}"
    echo "参数:"
    echo "  --force        强制终止服务进程"
    echo "  --help         显示此帮助信息"
    exit 1
}

# 默认参数
FORCE=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_usage
            ;;
    esac
done

# 检查服务是否在运行
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}警告: 找不到PID文件，服务可能未在运行${NC}"
    echo "如需启动服务，请运行 ./start.sh"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null; then
    echo -e "${YELLOW}警告: 服务进程 (PID: $PID) 未运行，将清理PID文件${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

# 停止服务
echo -e "${GREEN}正在停止 $SERVICE_NAME (PID: $PID)...${NC}"

if [ $FORCE -eq 1 ]; then
    # 强制终止
    echo "使用强制终止模式..."
    kill -9 $PID
else
    # 正常终止
    kill $PID
    
    # 等待服务停止
    echo "等待服务停止..."
    TIMEOUT=30
    for i in $(seq 1 $TIMEOUT); do
        if ! ps -p "$PID" > /dev/null; then
            break
        fi
        echo -n "."
        sleep 1
        
        # 如果超时，询问是否强制终止
        if [ $i -eq $TIMEOUT ]; then
            echo
            echo -e "${YELLOW}警告: 服务在 ${TIMEOUT} 秒内未停止${NC}"
            echo -n "是否强制终止进程? [y/N] "
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                echo "正在强制终止进程..."
                kill -9 $PID
            else
                echo "取消操作，服务将继续运行"
                exit 0
            fi
        fi
    done
    echo
fi

# 验证服务是否已停止
if ps -p "$PID" > /dev/null; then
    echo -e "${RED}错误: 无法停止服务，进程 (PID: $PID) 仍在运行${NC}"
    echo "请尝试使用 --force 参数强制终止"
    exit 1
else
    echo -e "${GREEN}$SERVICE_NAME 已成功停止${NC}"
    # 删除PID文件
    rm -f "$PID_FILE"
fi

# 同时终止所有相关进程
echo "正在检查并终止相关进程..."
RELATED_PIDS=$(ps -ef | grep "python run.py" | grep -v grep | awk '{print $2}')
if [ -n "$RELATED_PIDS" ]; then
    echo "发现相关进程: $RELATED_PIDS"
    for pid in $RELATED_PIDS; do
        if [ "$pid" != "$PID" ]; then
            echo "正在终止进程 PID: $pid"
            kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null
        fi
    done
    echo "相关进程已终止"
else
    echo "未发现其他相关进程"
fi

exit 0 