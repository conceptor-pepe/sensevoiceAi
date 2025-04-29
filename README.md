# SenseVoice API - 轻量级语音识别服务

## 项目简介

SenseVoice API 是一个基于 FastAPI 的轻量级语音识别服务，使用内存队列实现了异步任务处理，提供了简洁的 API 接口进行语音识别。该服务专为 SenseVoice 模型设计，支持高效的语音转文本能力。

## 功能特点

- 基于 FastAPI 的同步 API 接口，内部异步处理
- 内存队列实现，无需额外依赖 Redis 等外部服务
- 多工作器并行处理，提高吞吐量
- 自动管理任务生命周期
- 简单易用的 API 接口

## 环境要求

- Python 3.8+
- SenseVoice 模型
- CUDA 支持（用于模型推理，推荐）
- PyTorch 和 TorchAudio

## 安装步骤

1. 克隆代码仓库
```bash
git clone <仓库地址>
cd sensevoice-api
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 配置环境变量

可以通过环境变量自定义服务配置：

- `SENSEVOICE_MODEL_DIR`: SenseVoice 模型目录，默认为 "iic/SenseVoiceSmall"
- `SENSEVOICE_DEVICE`: 使用的设备，默认为 "cuda:0"
- `SENSEVOICE_WORKERS`: 工作器数量，默认为 4
- `SENSEVOICE_TIMEOUT`: 任务超时时间(秒)，默认为 30
- `SENSEVOICE_MAX_QUEUE`: 最大队列长度，默认为 100

### 启动服务

使用提供的启动脚本：

```bash
# 使用默认配置
./start.sh

# 指定设备和工作器数量
./start.sh --device cuda:1 --workers 8
```

或者直接启动：

```bash
# 启动服务
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API 接口说明

### 语音识别接口

```
POST /asr
```

参数：
- `files`: 音频文件列表 (multipart/form-data)
- `language`: 语言（可选，默认为 "auto"）

返回：任务ID
```json
{
  "task_id": "任务唯一标识符"
}
```

### 获取结果接口

```
GET /result/{task_id}
```

参数：
- `task_id`: 任务ID（路径参数）

返回：识别结果
```json
{
  "text": "识别出的文本",
  "task_id": "任务唯一标识符"
}
```

## 工作原理

1. **初始化阶段**：
   - 创建多个工作器线程，共享内存存储
   - 根据配置分配 GPU 资源

2. **处理流程**：
   - 客户端发送请求到 `/asr` 接口
   - API 服务将任务添加到内存队列并返回任务ID
   - 工作器从队列获取任务并处理
   - 工作器将语音识别结果存入内存存储
   - 客户端通过 `/result/{task_id}` 接口获取结果

3. **资源管理**：
   - 任务结果在指定时间后自动清理
   - 任务队列大小可配置，防止内存溢出

## 使用示例

### Python 客户端

```python
import requests
import time

# 提交语音识别请求
url = "http://localhost:8000/asr"
files = [
    ('files', ('audio1.wav', open('audio1.wav', 'rb'), 'audio/wav'))
]
data = {
    'language': 'auto'
}

response = requests.post(url, files=files, data=data)
task_id = response.json()["task_id"]

# 获取结果
result_url = f"http://localhost:8000/result/{task_id}"
while True:
    response = requests.get(result_url)
    if response.status_code == 200:
        result = response.json()
        print(f"识别结果: {result['text']}")
        break
    elif response.status_code == 404:
        print("等待结果...")
        time.sleep(1)
    else:
        print(f"错误: {response.status_code}")
        break
```

### cURL 示例

```bash
# 提交语音识别请求
TASK_ID=$(curl -s -X POST "http://localhost:8000/asr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@audio1.wav" \
  -F "language=auto" | jq -r '.task_id')

echo "任务ID: $TASK_ID"

# 获取结果
curl -X GET "http://localhost:8000/result/$TASK_ID" \
  -H "accept: application/json"
```

## 性能优化

1. **增加工作器数量**：
   通过 `SENSEVOICE_WORKERS` 环境变量调整，建议根据 GPU 内存和 CPU 核心数设置

2. **调整队列大小**：
   通过 `SENSEVOICE_MAX_QUEUE` 环境变量调整最大队列长度，需根据内存大小和预期负载设置

3. **使用更快的设备**：
   设置 `SENSEVOICE_DEVICE=cuda:0` 使用 GPU 加速，提高处理速度

## 常见问题

1. **服务启动失败**：
   - 确认模型目录路径正确
   - 检查 CUDA 环境和 GPU 可用性

2. **任务超时问题**：
   - 增加 `SENSEVOICE_TIMEOUT` 值
   - 减少工作器数量以减轻负载

3. **内存使用过高**：
   - 减小 `SENSEVOICE_MAX_QUEUE` 值
   - 减少并发请求数量