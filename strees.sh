#!/bin/bash
#!/bin/bash

# 压测配置
API_URL="http://222.211.217.10:8000/transcribe"    # API地址
FILES=("en.mp3" "zh.mp3")                     # 测试文件数组
THREADS=10                                    # 并发线程数
LOOPS=100                                     # 每个线程循环次数

# 日志记录
LOG_FILE="asr_stress_test_$(date +%Y%m%d%H%M%S).log"
echo "语音识别压力测试开始 $(date)" | tee $LOG_FILE
echo "配置: 并发数=$THREADS, 总请求数=$((THREADS*LOOPS))" | tee -a $LOG_FILE

# 压测函数
stress_test() {
    local thread_id=$1
    
    for ((i=1; i<=$LOOPS; i++)); do
        # 准备文件和键名
        FILE_PARAMS=""
        KEY_PARAMS=""
        
        # 构建文件参数和键名参数
        for FILE in "${FILES[@]}"; do
            FILE_PARAMS="$FILE_PARAMS -F \"files=@$FILE\""
            KEY_BASENAME=$(basename "$FILE" .mp3)
            if [ -z "$KEY_PARAMS" ]; then
                KEY_PARAMS="$KEY_BASENAME"
            else
                KEY_PARAMS="$KEY_PARAMS,$KEY_BASENAME"
            fi
        done
        
        START_TIME=$(date +%s.%N)
        
        # 执行请求并捕获输出
        CURL_CMD="curl -X POST \"$API_URL\" $FILE_PARAMS -F \"keys=$KEY_PARAMS\" -F \"lang=auto\" -w \"总耗时: %{time_total}秒\n\" -s -o /dev/null"
        RESPONSE=$(eval $CURL_CMD)
        
        END_TIME=$(date +%s.%N)
        ELAPSED=$(printf "%.3f" $(echo "$END_TIME - $START_TIME" | bc))
        
        # 记录日志
        echo "[线程 $thread_id] 第$i次请求 | 文件: ${FILES[*]} | $RESPONSE | 实际耗时: ${ELAPSED}秒" | tee -a $LOG_FILE
    done
}

# 启动多线程压测
echo "启动压测..." | tee -a $LOG_FILE
for ((t=1; t<=$THREADS; t++)); do
    stress_test $t &
done

# 等待所有后台任务完成
wait
echo "压测完成 $(date)" | tee -a $LOG_FILE
echo "详细日志见: $LOG_FILE"
