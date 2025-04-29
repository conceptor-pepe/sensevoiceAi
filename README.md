# SenseVoice API

SenseVoice API是一个高性能的语音识别服务，支持多语言识别，提供类似OpenAI的API接口，支持高并发处理和缓存优化。该项目是对原始[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)的高性能API封装。

## 功能特点

- **高性能推理**：优化模型加载和推理过程，提高处理速度
- **高并发支持**：并发处理多个请求，提高吞吐量
- **多语言支持**：支持中文、英文、粤语、日语、韩语等多种语言
- **兼容性API**：提供与OpenAI类似的API接口，便于集成
- **结果缓存**：自动缓存识别结果，提高重复请求的响应速度
- **多种输出格式**：支持JSON、纯文本、SRT、WebVTT等多种输出格式
- **系统监控**：实时监控系统资源和API性能

## 与原始SenseVoice的关系

本项目是对原始[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)的扩展和优化：

1. 使用FunASR提供的模型接口，集成SenseVoice模型
2. 提供高性能、高并发的API服务层
3. 添加缓存、批处理、监控等企业级功能
4. 提供友好的部署和使用工具

## 系统架构

本服务采用了基于FastAPI的高性能架构设计，主要组件包括：

- **FastAPI**: Web框架和API接口
- **FunASR**: SenseVoice模型的封装库
- **Redis**: 用于结果缓存
- **监控系统**: 系统资源和API性能监控

## 安装步骤

### 依赖要求

- Python 3.10 或更高版本
- CUDA支持的GPU（推荐）或CPU
- Redis服务器（可选，用于缓存）

### 安装流程

1. 克隆仓库

```bash
git clone https://github.com/your-username/sensevoice-api.git
cd sensevoice-api
```

2. 运行安装脚本

```bash
chmod +x install.sh
./install.sh
```

安装脚本会自动：
- 创建Python虚拟环境
- 安装所需依赖
- 下载并初始化SenseVoice模型
- 测试安装是否成功

3. 配置服务

可以创建一个`.env`文件在项目根目录，或者使用环境变量来配置服务：

```
# API基础配置
APP_NAME=SenseVoice API
APP_VERSION=1.0.0

# GPU配置
DEVICE=cuda:0  # 根据实际GPU情况调整

# 模型配置
MODEL_DIR=iic/SenseVoiceSmall

# 服务器配置 
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Redis配置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 安全配置（可选）
API_KEY_ENABLED=false
API_KEY=your_api_key_here
```

## 启动服务

```bash
# 基本启动
./start.sh

# 自定义参数启动
./start.sh --port 9000 --device cuda:1 --workers 8
```

## API使用指南

### 1. 语音转录（类OpenAI）

**端点**: `/api/v1/audio/transcriptions`

**请求方式**: POST

**请求参数**:
- `file`: 音频文件（支持WAV、MP3、OGG、WEBM）
- `model`: 模型ID（默认：sense-voice-small）
- `language`: 语言（auto, zh, en, yue, ja, ko）
- `response_format`: 响应格式（json, text, srt, vtt）

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "model=sense-voice-small" \
  -F "language=zh" \
  -F "response_format=json"
```

**响应示例**:

```json
{
  "result": [
    {
      "key": "audio.mp3",
      "text": "这是识别出的文本内容",
      "raw_text": "这是原始识别文本",
      "language": "zh"
    }
  ]
}
```

### 2. 流式语音转录

**端点**: `/api/v1/audio/transcriptions/stream`

**请求方式**: POST

**请求参数**:
- 请求体需包含音频流
- `model`: 模型ID
- `language`: 语言代码

**响应**:
流式返回JSON对象

### 3. 传统API（兼容旧版）

**端点**: `/api/v1/asr`

**请求方式**: POST

**请求参数**:
- `files`: 音频文件列表
- `keys`: 音频名称（逗号分隔）
- `lang`: 语言代码

### 4. 获取模型列表

**端点**: `/api/v1/models`

**请求方式**: GET

**响应示例**:

```json
{
  "models": [
    {
      "id": "sense-voice-small",
      "object": "model",
      "created": 1677610602,
      "owned_by": "iic"
    }
  ]
}
```

### 5. 监控与管理接口

#### 健康检查

**端点**: `/api/v1/health`

**请求方式**: GET

#### 性能统计

**端点**: `/api/v1/performance`

**请求方式**: GET

#### 系统指标

**端点**: `/api/v1/system/metrics`

**请求方式**: GET

## 客户端使用

我们提供了一个示例客户端脚本，方便您集成和测试API：

```bash
# 转录单个音频文件
python example_client.py --action transcribe --file audio.mp3 --language zh

# 批量转录
python example_client.py --action batch --files audio1.mp3 audio2.mp3 --language auto

# 创建SRT字幕
python example_client.py --action srt --file video.mp3 --output video.srt --language en
```

查看 [client_usage.md](client_usage.md) 获取更多客户端使用信息。

## 性能优化

本服务在以下方面进行了性能优化：

1. **模型优化**：
   - 支持ONNX和Torch两种模型格式
   - ONNX模型支持量化，减少内存占用
   - 批处理支持，提高吞吐量

2. **服务优化**：
   - 异步处理：使用FastAPI的异步特性提高并发能力
   - 并发控制：控制最大并发请求数
   - 请求队列：实现请求队列和优先级处理

3. **缓存优化**：
   - Redis缓存：缓存识别结果
   - 避免重复处理：通过音频指纹实现缓存

4. **监控和日志**：
   - 性能指标：监控API性能和处理时间
   - 系统监控：监控CPU、内存、GPU资源使用
   - 详细日志：记录处理过程和错误信息

## 管理工具

项目提供了多个管理脚本：

- `start.sh`: 启动服务
- `stop.sh`: 停止服务
- `status.sh`: 查看服务状态
- `init_model.py`: 初始化模型

查看帮助信息：

```bash
./start.sh --help
./status.sh --verbose
```

## 常见问题与解决

1. **模型下载失败**：
   - 检查网络连接
   - 尝试手动下载模型并放置在正确位置

2. **GPU内存不足**：
   - 减小批处理大小
   - 使用量化模型降低内存占用

3. **高并发处理**：
   - 调整`MAX_CONCURRENT_REQUESTS`参数
   - 增加Redis缓存大小

## 开发与贡献

欢迎贡献代码和反馈问题。开发时建议使用以下命令启动开发模式：

```bash
python run.py --reload --log-level debug
```

## 许可证

[MIT License](LICENSE)

## 致谢

本项目基于[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)和[FunASR](https://github.com/alibaba/FunASR)，感谢他们提供的优秀模型和工具。