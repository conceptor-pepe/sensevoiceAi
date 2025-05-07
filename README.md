# SenseVoice ASR API 服务

基于SenseVoiceSmall模型的高性能语音识别API服务，支持多种音频格式，提供RESTful接口。

## 功能特点

- **多格式音频支持**: 自动处理WAV, MP3, AAC, AMR, FLAC, OGG, OPUS, M4A, WEBM, WMA格式
- **自动音频处理**: 智能重采样、单声道转换，无需预处理
- **高性能GPU加速**: 支持CUDA加速，优化显存使用，提高推理速度
- **系统服务集成**: 作为systemd服务运行，实现开机自启和故障自恢复
- **完整性能监控**: 记录各阶段处理时间和资源使用，便于性能分析
- **完整API文档**: 集成Swagger文档，方便开发集成
- **结构化日志系统**: 详细记录操作日志和错误信息，方便问题定位

## 系统架构

### 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  客户端应用     │───>│  SenseVoice API │───>│  模型推理引擎   │
│  (Web/移动应用) │    │  (FastAPI服务)  │    │  (ONNX运行时)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              │                        │
                              ▼                        ▼
                       ┌─────────────┐         ┌─────────────────┐
                       │             │         │                 │
                       │ 音频处理器  │         │   性能监控      │
                       │             │         │                 │
                       └─────────────┘         └─────────────────┘
```

### 核心组件

- **API服务**: FastAPI应用，提供RESTful接口
- **模型管理器**: 负责模型加载、推理和资源管理
- **音频处理器**: 处理各种格式的音频输入
- **性能监控**: 记录各组件性能指标
- **日志系统**: 结构化日志记录和错误追踪

## 系统要求

- Python 3.8+
- NVIDIA GPU (CUDA 11.0+)
- 至少8GB显存
- Linux系统 (推荐Ubuntu 20.04+)

## 快速安装

### 自动安装

使用提供的安装脚本进行一键安装:

```bash
sudo bash install.sh
```

安装完成后，服务将自动启动并注册为系统服务。

### 手动安装

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 创建日志目录:

```bash
sudo mkdir -p /var/log/sensevoice
sudo chmod 755 /var/log/sensevoice
```

3. 配置服务:

```bash
sudo cp senseaudio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable senseaudio
sudo systemctl start senseaudio
```

## 使用方法

### API接口示例

#### 1. 音频转写接口

```bash
# 使用curl发送POST请求转写音频
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@/path/to/audio.mp3" \
  -F "language=auto"
```

**响应示例:**

```json
{
  "status": "success",
  "text": "这是识别出的文本内容",
  "processing_time": 1.234,
  "device": "cuda:5"
}
```

#### 2. 批量转写接口

```bash
# 使用curl发送POST请求批量转写多个音频
curl -X POST "http://localhost:8000/api/v1/asr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/audio1.mp3" \
  -F "files=@/path/to/audio2.mp3" \
  -F "keys=audio1,audio2" \
  -F "lang=auto"
```

#### 3. 服务状态查询

```bash
# 使用curl获取服务状态
curl -X GET "http://localhost:8000/status"
```

**响应示例:**

```json
{
  "status": "running",
  "version": "1.4",
  "gpu_status": {
    "available": true,
    "device_count": 1,
    "current_device": "cuda:5",
    "device_name": "NVIDIA RTX A6000"
  },
  "metrics": {
    "audio_processing_time": {
      "count": 10,
      "total": 2.345,
      "min": 0.123,
      "max": 0.456,
      "avg": 0.2345
    },
    "transcribe_time": {
      "count": 10,
      "total": 5.678,
      "min": 0.456,
      "max": 0.789,
      "avg": 0.5678
    }
  }
}
```

### 命令行参数

通过命令行参数可以自定义服务配置:

```bash
python main.py --help
```

可用参数:

- `--host`: 服务监听地址 (默认: 0.0.0.0)
- `--port`: 服务端口号 (默认: 8000)
- `--workers`: Worker进程数 (默认: 1)
- `--gpu`: 使用的GPU设备ID (默认: 5)
- `--debug`: 是否启用调试模式

## 项目结构

```
.
├── api.py              # API接口定义
├── audio_processor.py  # 音频处理模块
├── config.py           # 配置文件
├── logger.py           # 日志和性能监控
├── main.py             # 主程序入口
├── model_manager.py    # 模型管理器
├── model.py            # 模型定义
├── requirements.txt    # 依赖管理
├── install.sh          # 安装脚本
└── senseaudio.service  # systemd服务配置
```

## 性能优化

- **GPU内存优化**: 自动计算并限制GPU显存使用量
- **ONNX运行时配置**: 针对GPU优化的执行提供者配置
- **多线程控制**: 通过环境变量控制线程数量
- **TensorRT加速**: 支持FP16推理加速
- **并发处理**: 使用FastAPI的异步能力提高并发性能

## 日志与监控

系统提供了完整的日志记录和性能监控能力：

- **结构化日志**: 所有日志记录采用结构化格式，包含时间戳、日志级别、进程/线程信息
- **多级日志**: 分为主日志、错误日志和访问日志，便于问题追踪
- **日志轮转**: 支持基于时间的日志轮转和保留策略，默认每天轮转，保留7天
- **性能指标**: 记录各阶段处理时间，便于性能分析和问题定位

### 日志目录结构

所有日志存放在 `/var/log/sensevoice` 目录下：

- `senseaudio.log` - 主日志文件，记录所有日志信息
- `senseaudio_error.log` - 错误日志，仅记录ERROR及以上级别的信息
- `senseaudio_access.log` - 访问日志，记录API访问信息

### 查看日志

```bash
# 查看主日志
sudo tail -f /var/log/sensevoice/senseaudio.log

# 查看错误日志
sudo tail -f /var/log/sensevoice/senseaudio_error.log

# 查看访问日志
sudo tail -f /var/log/sensevoice/senseaudio_access.log

# 查看系统日志
journalctl -u senseaudio -f
```

## 开发者扩展

如需扩展功能:

1. 修改 `config.py` 文件更新配置
2. 在 `api.py` 中添加新的API端点
3. 重启服务使更改生效:

```bash
sudo systemctl restart senseaudio
```

## 处理流程详解

当用户发送请求时，系统按以下步骤处理：

1. **请求接收与验证**
   - FastAPI接收POST请求到`/transcribe`端点
   - 验证音频文件格式是否在支持列表中
   - 格式无效则返回400错误

2. **音频处理**
   - 读取音频内容
   - 通过音频处理器进行处理：
     * 重采样到16kHz
     * 转换为单声道（如果是多声道）
   - 将处理后的音频转换为张量

3. **语音识别**
   - 调用模型进行推理
   - 获取识别文本结果

4. **结果处理**
   - 构建包含文本结果、处理时间等信息的JSON响应
   - 记录性能指标
   - 将结果返回给用户

5. **全程性能监控**
   - 各处理阶段的时间被记录到PerformanceMonitor
   - 整个函数执行时间由timer装饰器记录
   - 异常处理机制确保错误被记录并向用户返回适当响应