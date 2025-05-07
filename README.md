# SenseVoice ASR API 服务

基于SenseVoiceSmall模型的高性能语音识别API服务，支持多种音频格式，提供RESTful接口。

## 功能特点

- **支持多种音频格式**: WAV, MP3, AAC, AMR, FLAC, OGG, OPUS, M4A, WEBM, WMA
- **自动音频处理**: 自动重采样、单声道转换
- **高性能GPU加速**: 支持CUDA加速，优化显存使用
- **结果缓存**: 对相同音频请求进行缓存，提高响应速度
- **系统服务集成**: 支持作为systemd服务运行
- **详细性能指标**: 记录各阶段处理时间，便于性能分析
- **完整API文档**: 自动生成的Swagger文档

## 系统要求

- Python 3.8+
- NVIDIA GPU (CUDA 11.0+)
- 至少8GB显存

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

2. 配置服务:

```bash
sudo cp senseaudio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable senseaudio
sudo systemctl start senseaudio
```

## 使用方法

### API接口

服务启动后，API接口可通过 `http://localhost:8000` 访问。

#### 1. 音频转写接口

```
POST /transcribe
```

**参数:**

- `audio`: 音频文件 (multipart/form-data)
- `language`: 语言设置 (auto/zh/en/ja...)，默认为auto
- `textnorm`: 文本规范化设置 (withitn/noitn)，默认为withitn

**示例请求:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@/path/to/audio.mp3" \
  -F "language=auto"
```

**响应:**

```json
{
  "status": "success",
  "text": "这是识别出的文本内容",
  "cached": false,
  "processing_time": 1.234,
  "sample_rate": "16000Hz",
  "device": "GPU5(NVIDIA RTX A6000)"
}
```

#### 2. 服务状态接口

```
GET /status
```

**响应:**

```json
{
  "status": "running",
  "version": "1.3",
  "gpu_status": {
    "status": "available",
    "device_count": 1,
    "devices": [
      {
        "id": 0,
        "name": "NVIDIA RTX A6000",
        "memory_total": 51539607552,
        "memory_allocated": 1073741824,
        "memory_reserved": 2147483648
      }
    ],
    "current_device": "cuda:0"
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
- `--cache`: 是否启用缓存 (默认: True)
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
├── requirements.txt    # 依赖管理
├── install.sh          # 安装脚本
└── senseaudio.service  # systemd服务配置
```

## 性能优化

- **显存优化**: 自动计算并限制GPU显存使用量
- **缓存机制**: 对处理过的音频结果进行缓存
- **后台任务**: 使用后台任务清理临时文件
- **模型单例**: 确保只加载一个模型实例

## 日志与监控

服务运行日志存储在 `logs/senseaudio.log`，同时也会输出到系统日志。

通过系统日志查看服务状态:

```bash
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

## 详细说明文档

### SenseVoice API处理流程

当用户发送请求（如`curl -X POST -F "audio=@./zh.mp3" http://localhost:8000/transcribe`）时，系统按以下步骤处理：

1. **请求接收与验证**
   - FastAPI接收POST请求到`/transcribe`端点
   - 验证音频文件格式是否在支持列表中（WAV, MP3, AAC等）
   - 格式无效则返回400错误

2. **缓存检查**
   - 读取音频内容并计算唯一MD5哈希值
   - 使用哈希值查询缓存系统
   - 如果缓存命中，直接返回缓存结果，跳过处理步骤

3. **音频处理**（缓存未命中时）
   - 将原始音频保存为临时文件
   - 通过librosa进行音频处理：
     * 重采样到16kHz（标准采样率）
     * 转换为单声道（如果是多声道）
   - 处理后的音频保存为临时WAV文件

4. **语音识别**
   - 调用ModelManager中的transcribe方法
   - 使用SenseVoiceSmall模型进行语音识别
   - 获取识别文本结果

5. **结果处理**
   - 构建包含文本结果、处理时间等信息的JSON响应
   - 将结果添加到缓存系统（使用之前的哈希值）
   - 安排后台任务清理临时文件
   - 将结果返回给用户

6. **全程性能监控**
   - 各处理阶段的时间被记录到PerformanceMonitor
   - 整个函数执行时间由timer装饰器记录
   - 异常处理机制确保错误被记录并向用户返回适当响应

### 架构特点

- **模块化设计**：处理流程分散在不同模块，职责明确
- **异步处理**：使用FastAPI的异步能力提高并发性
- **缓存机制**：通过音频指纹避免重复处理相同音频
- **资源管理**：后台任务确保临时文件被正确清理
- **性能监控**：详细记录各阶段处理时间，便于优化

image.png

## 性能优化说明

### 最新优化

本次优化旨在提高API服务的性能，主要包括以下方面：

1. **直接使用ONNX推理**
   - 使用更高效的ONNX运行时加速推理
   - 原有架构：ModelManager -> SenseVoiceSmall
   - 优化架构：直接使用SenseVoiceSmall

2. **简化模型加载与初始化**
   - 移除了多层封装和依赖注入
   - 全局单例模型初始化，避免重复加载
   - 减少不必要的日志和检查

3. **添加兼容API接口**
   - 添加`/api/v1/asr`端点支持批量处理
   - 自定义结果格式，提供原始和处理后的文本
   - 兼容其他服务的API调用方式

4. **内存优化**
   - 使用BytesIO处理内存中的数据
   - 减少不必要的数据转换
   - 添加异常处理和资源清理

5. **环境变量控制**
   - 支持通过`SENSEVOICE_DEVICE`环境变量切换GPU设备
   - 自动配置CUDA可见设备

### 性能结果

- 原有版本：约5秒处理时间
- 优化版本：约1秒处理时间

## 许可证

[Apache 2.0](LICENSE)