#!/bin/bash
# SenseVoice API 状态检查脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 服务名称
SERVICE_NAME="SenseVoice API"
PID_FILE="sensevoice_api.pid"
LOG_DIR="logs"

# 显示使用方法
show_usage() {
    echo -e "${YELLOW}使用方法: $0 [参数]${NC}"
    echo "参数:"
    echo "  --verbose      显示详细信息"
    echo "  --help         显示此帮助信息"
    exit 1
}

# 默认参数
VERBOSE=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose)
            VERBOSE=1
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

# 打印分隔线
print_separator() {
    echo -e "${BLUE}------------------------------------------------${NC}"
}

# 检查服务状态
check_service_status() {
    # 检查 PID 文件
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}服务状态: 未运行${NC}"
        echo "未发现 PID 文件，服务可能未启动"
        echo "使用 ./start.sh 启动服务"
        return 1
    fi

    PID=$(cat "$PID_FILE")

    # 检查进程是否存在
    if ps -p "$PID" > /dev/null; then
        echo -e "${GREEN}服务状态: 运行中${NC}"
        echo "进程 ID: $PID"
        
        # 获取进程运行时间
        if command -v ps >/dev/null 2>&1; then
            PROCESS_START=$(ps -p "$PID" -o lstart= 2>/dev/null)
            if [ -n "$PROCESS_START" ]; then
                echo "启动时间: $PROCESS_START"
            fi
            
            # 获取CPU和内存使用情况
            CPU_USAGE=$(ps -p "$PID" -o %cpu= 2>/dev/null)
            MEM_USAGE=$(ps -p "$PID" -o %mem= 2>/dev/null)
            if [ -n "$CPU_USAGE" ] && [ -n "$MEM_USAGE" ]; then
                echo "CPU 使用率: $CPU_USAGE%"
                echo "内存使用率: $MEM_USAGE%"
            fi
        fi
        
        return 0
    else
        echo -e "${RED}服务状态: 异常${NC}"
        echo "PID 文件存在 ($PID_FILE)，但进程 $PID 不存在"
        echo "可能服务异常终止，建议清理 PID 文件并重新启动服务"
        echo "使用以下命令清理:"
        echo "  rm -f $PID_FILE"
        echo "  ./start.sh"
        return 1
    fi
}

# 获取日志信息
get_log_info() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "日志目录不存在: $LOG_DIR"
        return
    fi
    
    echo "日志文件:"
    find "$LOG_DIR" -type f -name "*.log" -o -name "nohup.out" | sort | while read -r log_file; do
        if [ -f "$log_file" ]; then
            SIZE=$(du -h "$log_file" | cut -f1)
            MODIFIED=$(stat -c %y "$log_file" 2>/dev/null || stat -f "%Sm" "$log_file" 2>/dev/null)
            echo "  $log_file ($SIZE, 最后修改: $MODIFIED)"
            
            # 获取最近的日志
            if [ "$VERBOSE" -eq 1 ]; then
                echo "  最近日志:"
                tail -n 5 "$log_file" | sed 's/^/    /'
                echo
            fi
        fi
    done
}

# 检查API可用性
check_api_availability() {
    # 首先检查服务是否在运行
    if [ ! -f "$PID_FILE" ]; then
        return
    fi
    
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null; then
        return
    fi
    
    # 尝试获取端口
    PORT=$(ps -ef | grep "$PID" | grep -o "\-\-port [0-9]*" | awk '{print $2}')
    if [ -z "$PORT" ]; then
        PORT="8000"  # 默认端口
    fi
    
    # 检查API健康端点
    echo "正在检查API可用性..."
    if command -v curl >/dev/null 2>&1; then
        HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/api/v1/health 2>/dev/null)
        if [ "$HEALTH_RESPONSE" = "200" ]; then
            echo -e "${GREEN}API健康状态: 可用${NC}"
            echo "API端点: http://localhost:$PORT"
            echo "API文档: http://localhost:$PORT/docs"
        else
            echo -e "${RED}API健康状态: 不可用 (HTTP状态码: $HEALTH_RESPONSE)${NC}"
            echo "API端点可能未就绪或服务启动中"
        fi
    else
        echo "curl命令不可用，无法检查API健康状态"
    fi
}

# 主函数
main() {
    echo -e "${BLUE}=$SERVICE_NAME 状态报告=${NC}"
    print_separator
    
    # 检查服务状态
    echo "【服务状态】"
    check_service_status
    SERVICE_RUNNING=$?
    print_separator
    
    # 获取日志信息
    echo "【日志信息】"
    get_log_info
    print_separator
    
    # 如果服务在运行，检查API可用性
    if [ $SERVICE_RUNNING -eq 0 ]; then
        echo "【API状态】"
        check_api_availability
        print_separator
    fi
    
    # 如果开启详细模式，显示系统资源
    if [ "$VERBOSE" -eq 1 ]; then
        echo "【系统资源】"
        echo "CPU使用率:"
        top -bn1 | grep "Cpu(s)" | sed 's/.*, *\([0-9.]*\)%* id.*/\1/' | awk '{print 100 - $1"%"}'
        
        echo "内存使用率:"
        free -h | grep Mem | awk '{print $3 " / " $2 " (" int($3/$2*100) "%)"}'
        
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "GPU信息:"
            nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  /'
        fi
        print_separator
    fi
}

# 运行主函数
main 