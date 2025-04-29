#!/bin/bash
# SenseVoice API 启动脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 服务名称
SERVICE_NAME="SenseVoice API"
PID_FILE="sensevoice_api.pid"
LOG_DIR="logs"

# 默认配置
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_WORKERS="4"
DEFAULT_DEVICE="cuda:5"
DEFAULT_LOG_LEVEL="info"

# 创建日志目录
mkdir -p $LOG_DIR

# 显示使用方法
show_usage() {
    echo -e "${YELLOW}使用方法: $0 [参数]${NC}"
    echo "参数:"
    echo "  --host <host>       监听主机名 (默认: $DEFAULT_HOST)"
    echo "  --port <port>       监听端口 (默认: $DEFAULT_PORT)"
    echo "  --workers <num>     工作进程数 (默认: $DEFAULT_WORKERS)"
    echo "  --device <device>   CUDA设备 (默认: $DEFAULT_DEVICE)"
    echo "  --log-level <level> 日志级别 [debug|info|warning|error|critical] (默认: $DEFAULT_LOG_LEVEL)"
    echo "  --no-monitor        禁用系统监控"
    echo "  --help              显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 --port 9000 --device cuda:0 --workers 8"
    exit 1
}

# 解析命令行参数
HOST=$DEFAULT_HOST
PORT=$DEFAULT_PORT
WORKERS=$DEFAULT_WORKERS
DEVICE=$DEFAULT_DEVICE
LOG_LEVEL=$DEFAULT_LOG_LEVEL
MONITOR_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --no-monitor)
            MONITOR_FLAG="--no-monitor"
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

# 检查服务是否已在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo -e "${RED}错误: $SERVICE_NAME 已在运行，PID: $PID${NC}"
        echo "如需停止服务，请运行 ./stop.sh"
        exit 1
    else
        echo -e "${YELLOW}警告: 发现过期的PID文件，将被删除${NC}"
        rm "$PID_FILE"
    fi
fi

# 构建启动命令
# python run.py --host 0.0.0.0 --port 8000 --workers 4 --device cuda:5
COMMAND="nohup python run.py --host $HOST --port $PORT --workers $WORKERS --device $DEVICE --log-level $LOG_LEVEL $MONITOR_FLAG > $LOG_DIR/nohup.out 2>&1 &"

# 显示启动信息
echo -e "${GREEN}正在启动 $SERVICE_NAME...${NC}"
echo "配置参数:"
echo "  主机名: $HOST"
echo "  端口: $PORT"
echo "  工作进程数: $WORKERS"
echo "  CUDA设备: $DEVICE"
echo "  日志级别: $LOG_LEVEL"
if [ -n "$MONITOR_FLAG" ]; then
    echo "  系统监控: 已禁用"
else
    echo "  系统监控: 已启用"
fi

# 执行启动命令
eval $COMMAND
PID=$!

# 保存PID
echo $PID > "$PID_FILE"

# 等待服务启动
sleep 2

# 检查服务是否成功启动
if ps -p "$PID" > /dev/null; then
    echo -e "${GREEN}$SERVICE_NAME 已成功启动，PID: $PID${NC}"
    echo "日志文件: $LOG_DIR/nohup.out"
    echo "API文档: http://$HOST:$PORT/docs"
else
    echo -e "${RED}错误: $SERVICE_NAME 启动失败${NC}"
    echo "请检查日志文件: $LOG_DIR/nohup.out"
    rm -f "$PID_FILE"
    exit 1
fi

exit 0 

