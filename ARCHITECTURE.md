# SenseVoice API 系统架构文档

## 1. 项目概述

SenseVoice API 是一个高性能的语音识别服务，基于FunASR框架开发，支持中英文语音识别。系统采用模块化设计，提供REST API接口，支持批量和流式语音识别，适用于各类语音转文字应用场景。

## 2. 目录结构与文件功能

### 2.1 核心组件

| 文件名 | 功能描述 |
|--------|---------|
| `api.py` | FastAPI主程序，提供HTTP接口服务 |
| `model.py` | 模型统一接口层，自动选择PyTorch或ONNX后端 |
| `model_manager.py` | 模型管理器，负责模型加载、推理和并发控制 |
| `init_model.py` | 模型初始化脚本，下载和验证模型 |
| `config.py` | 系统配置管理，包含所有可配置项 |
| `run.py` | 服务程序入口，处理命令行参数并启动API服务 |

### 2.2 工具和辅助模块

| 文件名 | 功能描述 |
|--------|---------|
| `cache.py` | 识别结果缓存管理，提高重复请求性能 |
| `performance.py` | 性能监控和统计，收集API响应时间等指标 |
| `monitoring.py` | 系统监控，包括CPU、内存、GPU利用率监控 |
| `utils.py` | 通用工具函数集合 |
| `logging_config.py` | 日志系统配置 |

### 2.3 脚本和部署文件

| 文件名 | 功能描述 |
|--------|---------|
| `install.sh` | 系统安装脚本，自动安装依赖并初始化环境 |
| `start.sh` | 启动服务脚本 |
| `stop.sh` | 停止服务脚本 |
| `status.sh` | 查看服务状态脚本 |
| `requirements.txt` | 项目依赖列表 |
| `settings.env` | 环境变量配置文件 |

### 2.4 文档文件

| 文件名 | 功能描述 |
|--------|---------|
| `README.md` | 项目总体说明文档 |
| `api_documentation.md` | API详细文档 |
| `client_usage.md` | 客户端使用指南 |
| `ARCHITECTURE.md` | 本文档，系统架构说明 |

### 2.5 示例和客户端

| 文件名 | 功能描述 |
|--------|---------|
| `example_client.py` | 示例客户端程序，展示如何调用API |

## 3. 系统架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      客户端应用                              │
└───────────────────────────────┬─────────────────────────────┘
                                │ HTTP/WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 接口层 (api.py)                │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
┌───────────────▼─────────────┐   ┌─────────────▼───────────┐
│     缓存系统 (cache.py)     │   │  监控系统 (monitoring.py) │
└───────────────┬─────────────┘   └─────────────────────────┘
                │                               ▲
┌───────────────▼─────────────┐                 │
│  模型管理器 (model_manager.py)│────────────────┘
└───────────────┬─────────────┘   性能指标 (performance.py)
                │
┌───────────────▼─────────────┐
│   模型接口层 (model.py)      │
└───────────────┬─────────────┘
                │
        ┌───────┴────────┐
        │                │
┌───────▼───────┐  ┌─────▼──────┐
│ PyTorch 后端  │  │ ONNX 后端   │
└───────────────┘  └────────────┘
```

### 3.2 组件关系说明

1. **接口层 (api.py)**
   - 提供REST API接口，接收客户端请求
   - 处理请求参数验证和响应格式化
   - 调用模型管理器进行推理

2. **模型管理器 (model_manager.py)**
   - 管理模型的生命周期，包括加载和释放
   - 处理并发请求，控制系统负载
   - 协调请求处理流程

3. **模型接口层 (model.py)**
   - 提供统一的模型接口
   - 自动选择合适的后端实现(PyTorch/ONNX)
   - 简化上层对不同模型实现的调用方式

4. **缓存系统 (cache.py)**
   - 缓存识别结果，避免重复计算
   - 支持Redis和内存缓存两种方式
   - 提高系统整体吞吐量

5. **监控系统 (monitoring.py & performance.py)**
   - 收集系统资源指标，如CPU/GPU利用率
   - 统计API性能指标，如响应时间、错误率
   - 提供监控端点，支持Prometheus集成

## 4. 数据流程

### 4.1 请求处理流程

1. 客户端发送包含音频数据的POST请求到API接口
2. FastAPI接收请求并验证参数
3. 查询缓存系统，检查是否有缓存结果
4. 若无缓存，则调用模型管理器处理请求
5. 模型管理器通过模型接口调用适当的模型实现
6. 模型处理完成后返回识别结果
7. 将结果存入缓存并返回给客户端

### 4.2 模型初始化流程

1. 系统启动时，调用model_manager初始化模型
2. model_manager通过model.py统一接口创建模型实例
3. model.py尝试加载PyTorch版本模型，若失败则尝试ONNX版本
4. 模型加载完成后，进行一次测试推理确保功能正常
5. 系统进入就绪状态，可以接收请求

## 5. 模型管理

### 5.1 模型初始化

`init_model.py`脚本负责模型的初始化过程：

```python
# 初始化模型示例
python init_model.py --model-dir "iic/SenseVoiceSmall"
```

初始化过程包括：
- 检查环境和依赖
- 下载或加载模型文件
- 测试模型功能
- 配置模型参数

### 5.2 模型接口

`model.py`提供了统一的模型接口，屏蔽了底层实现差异：

```python
# 使用示例
from model import SenseVoiceSmall

# 创建模型实例
model = SenseVoiceSmall(model_dir="iic/SenseVoiceSmall", device="cuda:0")

# 进行语音识别
result = model.infer("audio.wav", language="auto")
print(result)
```

## 6. API接口

### 6.1 主要接口

| 接口路径 | 方法 | 功能描述 |
|---------|------|---------|
| `/api/v1/asr` | POST | 批量音频识别 |
| `/api/v1/asr/stream` | WebSocket | 流式音频识别 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/info` | GET | 获取系统信息 |
| `/api/v1/metrics` | GET | 获取监控指标 |

### 6.2 请求示例

**批量识别请求**:

```bash
curl -X POST "http://localhost:8000/api/v1/asr" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@audio.wav" \
  -F "language=auto"
```

**响应示例**:

```json
{
  "status": "success",
  "results": [
    {
      "text": "语音识别的文本结果",
      "confidence": 0.96,
      "language": "zh",
      "duration": 2.5
    }
  ],
  "processing_time": 0.324
}
```

## 7. 部署与运行

### 7.1 安装

```bash
# 安装方法
./install.sh
```

安装过程会：
- 创建Conda环境
- 安装依赖
- 初始化模型

### 7.2 启动服务

```bash
# 启动服务
./start.sh

# 或使用更多参数
./start.sh --port 8080 --device cuda:0 --workers 4
```

### 7.3 停止服务

```bash
# 停止服务
./stop.sh
```

## 8. 客户端使用示例

### 8.1 Python客户端

```python
import requests

def recognize_audio(audio_file, language="auto"):
    """
    调用SenseVoice API识别音频文件
    
    参数:
        audio_file: 音频文件路径
        language: 语言选择，auto为自动识别
        
    返回:
        识别结果字典
    """
    url = "http://localhost:8000/api/v1/asr"
    
    with open(audio_file, "rb") as f:
        files = {"audio_file": f}
        data = {"language": language}
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"请求失败: {response.status_code}, {response.text}")

# 使用示例
result = recognize_audio("test.wav")
print(result["results"][0]["text"])
```

### 8.2 流式识别示例

```python
import asyncio
import websockets
import json
import wave

async def stream_audio(audio_file, chunk_size=4096):
    """
    流式发送音频文件到SenseVoice API
    
    参数:
        audio_file: 音频文件路径
        chunk_size: 每个音频块的大小
    """
    uri = "ws://localhost:8000/api/v1/asr/stream"
    
    async with websockets.connect(uri) as websocket:
        # 发送配置
        await websocket.send(json.dumps({
            "language": "auto",
            "sample_rate": 16000
        }))
        
        # 读取音频并分块发送
        with wave.open(audio_file, "rb") as wav:
            while True:
                data = wav.readframes(chunk_size)
                if not data:
                    break
                await websocket.send(data)
            
            # 发送结束标记
            await websocket.send(json.dumps({"eof": True}))
        
        # 接收结果
        while True:
            result = await websocket.recv()
            result = json.loads(result)
            
            if result.get("final"):
                print("最终结果:", result["text"])
                break
            else:
                print("中间结果:", result["text"])

# 使用示例
asyncio.run(stream_audio("test.wav"))
```

## 9. 扩展与定制

### 9.1 添加新的模型

要添加新的模型类型:

1. 在`model.py`中创建新的模型接口
2. 在`model_manager.py`中添加对应的加载和处理逻辑
3. 更新`config.py`中的模型配置参数

### 9.2 性能优化

针对性能优化，可以:

- 调整`config.py`中的并发请求数
- 启用模型量化减少内存占用
- 配置适当的缓存策略
- 增加工作进程数量

## 10. 常见问题

### 10.1 故障排查

**问题**: 服务启动失败
**解决方案**: 
- 检查日志文件
- 确认GPU可用性
- 验证模型文件是否完整

**问题**: 推理速度慢
**解决方案**:
- 确认是否使用GPU
- 调整批处理大小
- 检查并发请求数配置

### 10.2 资源需求

- **最小配置**: 4GB内存，CPU推理
- **推荐配置**: 8GB内存，NVIDIA GPU 4GB+
- **存储需求**: 约2GB模型文件存储空间

---

文档编写日期: 2023年9月15日
版本: 1.0 