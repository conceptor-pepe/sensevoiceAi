# SenseVoice API 服务

基于SenseVoice Small模型的高性能语音识别API服务，支持多语言、多种情感识别和事件检测。

## 1. 功能介绍

### 1.1 核心功能

- **语音识别转写**：将语音内容转换为高质量文本
- **多语言支持**：自动识别并支持中文(zh)、英文(en)、粤语(yue)、日语(ja)、韩语(ko)
- **情感识别**：自动检测语音中表达的情绪，包括快乐(HAPPY)、悲伤(SAD)、愤怒(ANGRY)、中性(NEUTRAL)、恐惧(FEARFUL)、厌恶(DISGUSTED)和惊讶(SURPRISED)
- **事件检测**：识别语音中的背景音乐(BGM)、演讲(Speech)、掌声(Applause)、笑声(Laughter)、哭声(Cry)、打喷嚏(Sneeze)、呼吸(Breath)和咳嗽(Cough)等事件
- **流式识别**：支持HTTP流式响应和WebSocket实时交互，适用于实时语音识别场景

### 1.2 技术特点

- **ONNX运行时**：基于ONNX Runtime优化推理速度，充分利用硬件加速能力
- **GPU加速**：支持特定GPU设备指定，优化资源利用
- **批处理支持**：支持批量音频处理，提升吞吐量
- **灵活输入**：支持多种音频输入方式（文件上传和Base64编码）
- **高级文本处理**：支持反向文本归一化(ITN)，提升数字、日期等特殊内容的识别质量
- **可靠性设计**：内置进程管理和自动重启功能，保障服务可靠运行
- **流式处理**：支持音频文件分块流式处理和实时WebSocket流式识别，降低识别延迟

## 2. 架构设计

### 2.1 整体架构

SenseVoice API服务采用模块化设计，主要由以下组件构成：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     API层       │    │    处理层       │    │    模型层       │
│  (FastAPI框架)   │───▶│  (音频处理逻辑)  │───▶│ (SenseVoice模型)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   配置管理      │    │   时间统计      │    │   日志系统      │
│  (环境变量配置)  │    │ (性能监控分析)  │    │ (请求追踪记录)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 核心模块

- **API模块 (api.py)**：提供RESTful API接口，处理HTTP请求和响应
- **模型管理模块 (model_manager.py)**：负责模型加载和管理，实现单例模式避免重复加载
- **音频处理模块 (processor.py)**：处理音频文件、提取标签和格式化结果
- **配置模块 (config.py)**：集中管理系统配置和环境变量
- **统计模块 (stats.py)**：记录处理时间统计，帮助性能分析
- **日志模块 (logger.py)**：统一日志记录，支持请求追踪

### 2.3 数据流程

1. **请求接收**：API层接收HTTP请求，支持文件上传或Base64编码音频
2. **音频预处理**：处理层保存临时文件，准备推理数据
3. **模型推理**：调用SenseVoice Small模型进行语音识别
4. **后处理**：处理原始输出，提取文本、语言标签、情感标签和事件标签
5. **结果返回**：格式化JSON响应并返回给客户端
6. **资源清理**：自动清理临时文件，释放资源

### 2.4 服务管理

服务提供三层进程管理机制，确保高可用性：

1. **基础守护进程**：提供启动、停止、重启和状态查询功能
2. **监控与自动重启**：定期检查服务状态并自动恢复
3. **系统服务集成**：支持Systemd服务管理，实现开机自启动

## 3. 后期优化

### 3.1 性能优化

- **模型量化**：支持模型量化，减少内存占用并提升推理速度
- **批处理优化**：调整批处理大小参数，平衡延迟和吞吐量
- **设备选择**：支持指定GPU设备ID，充分利用多GPU环境
- **内存管理**：优化临时文件处理流程，减少内存占用

### 3.2 功能扩展

- **支持更多语言模型**：扩展语言支持范围
- **增强情感识别**：提升情感识别准确度和细粒度
- **自定义词典**：支持领域定制词汇，提升特定场景识别质量
- **多模型加载**：支持同时加载多个模型，根据场景自动选择最适合的模型

### 3.3 可靠性增强

- **健康检查机制**：定期检测模型状态，确保服务健康
- **限流保护**：添加请求限流功能，防止过载
- **负载均衡**：支持多实例部署和负载均衡
- **请求重试机制**：关键操作失败时自动重试，提高成功率
- **监控告警**：集成监控系统，异常情况及时告警

## 4. 使用介绍

### 4.1 环境要求

- Python 3.8+
- CUDA 11.0+（GPU加速）
- 足够的内存和磁盘空间用于模型加载

### 4.2 安装部署

#### 4.2.1 直接安装

```bash
# 克隆代码库
git clone https://github.com/yourusername/sensevoice-api.git
cd sensevoice-api

# 安装依赖
pip install -r requirements.txt

# 赋予启动脚本执行权限
chmod +x start.sh

# 使用默认配置启动
./start.sh

# 或者自定义配置
SENSEVOICE_GPU_DEVICE=1 SENSEVOICE_PORT=8080 ./start.sh
```

#### 4.2.2 Docker部署

```bash
# 构建Docker镜像
docker build -t sensevoice-api .

# 运行容器
docker run --gpus '"device=0"' -p 8000:8000 sensevoice-api

# 或者自定义配置
docker run --gpus '"device=1"' -p 8080:8000 \
  -e SENSEVOICE_GPU_DEVICE=1 \
  -e SENSEVOICE_PORT=8000 \
  -e SENSEVOICE_MODEL_DIR="iic/SenseVoiceSmall" \
  sensevoice-api
```

#### 4.2.3 系统服务管理
SenseVoice API 提供了 systemd 服务配置，可以作为系统服务运行：

```bash
# 赋予安装脚本执行权限
chmod +x install.sh

# 使用root权限运行安装脚本
sudo ./install.sh

# 安装脚本会完成以下操作：
# 1. 创建必要的日志和运行目录
# 2. 更新服务配置文件中的路径
# 3. 复制服务文件到系统目录
# 4. 创建配置目录和示例环境变量文件
# 5. 重新加载systemd配置
# 6. 启用服务（设置开机自启）
# 7. 询问是否立即启动服务

# 安装完成后，可以使用以下命令管理服务：
sudo systemctl start sensevoice.service    # 启动服务
sudo systemctl stop sensevoice.service     # 停止服务
sudo systemctl restart sensevoice.service  # 重启服务
sudo systemctl status sensevoice.service   # 查看服务状态
sudo journalctl -u sensevoice.service      # 查看服务日志
sudo systemctl enable sensevoice.service   # 启用服务（开机自启）
sudo systemctl disable sensevoice.service  # 禁用服务（关闭开机自启）

# 自定义环境变量配置：
# 1. 复制示例配置文件
sudo cp /etc/sensevoice/env.conf.example /etc/sensevoice/env.conf
# 2. 编辑配置文件
sudo nano /etc/sensevoice/env.conf
# 3. 取消注释并修改需要的环境变量
# 4. 编辑服务文件启用环境变量配置
sudo nano /etc/systemd/system/sensevoice.service
# 5. 取消注释 EnvironmentFile 行
# 6. 重新加载systemd并重启服务
sudo systemctl daemon-reload
sudo systemctl restart sensevoice.service
```

如果您不想将服务安装为系统服务，也可以使用项目中的`start.sh`脚本直接启动：

```bash
# 赋予启动脚本执行权限
chmod +x start.sh

# 使用默认配置启动
./start.sh

# 或者自定义环境变量启动
SENSEVOICE_GPU_DEVICE=1 SENSEVOICE_PORT=8080 ./start.sh
```

### 4.3 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| SENSEVOICE_MODEL_DIR | 模型目录 | iic/SenseVoiceSmall |
| SENSEVOICE_GPU_DEVICE | GPU设备ID | 0 |
| SENSEVOICE_HOST | 监听地址 | 0.0.0.0 |
| SENSEVOICE_PORT | 监听端口 | 8000 |
| SENSEVOICE_BATCH_SIZE | 批处理大小 | 1 |
| SENSEVOICE_LOG_LEVEL | 日志级别 | INFO |
| SENSEVOICE_LOG_FILE | 日志文件路径 | (默认输出到控制台) |
| SENSEVOICE_TEMP_DIR | 临时文件目录 | /tmp |
| SENSEVOICE_CHUNK_SIZE_SEC | 流式处理默认分块大小（秒） | 3 |
| SENSEVOICE_MAX_CLIENTS | WebSocket最大并发连接数 | 10 |

### 4.4 API接口

#### 4.4.1 健康检查

```
GET /
```

响应示例：

```json
{
  "status": "ok",
  "message": "SenseVoice API服务运行正常"
}
```

#### 4.4.2 语音识别 - 标准接口

```
POST /recognize
```

**支持两种请求方式**：

1. **表单上传**：直接上传音频文件

   请求参数：
   - audio_file: 音频文件（支持多种格式）
   - language: 语言代码（可选，默认"auto"）
   - use_itn: 是否使用ITN（可选，默认为true）

2. **JSON请求**：使用Base64编码音频

   请求参数：
   - request_data: JSON字符串，包含以下字段：
     - audio_base64: Base64编码的音频数据
     - language: 语言代码（可选，默认"auto"）
     - use_itn: 是否使用ITN（可选，默认为true）

响应示例：

```json
{
  "success": true,
  "message": "识别成功",
  "text": "识别出的文本内容",
  "language": "zh",
  "emotion": "NEUTRAL",
  "event": "Speech",
  "time_cost": 1.23,
  "detail_time": {
    "文件上传": 0.05,
    "数据准备": 0.1,
    "模型推理": 1.0,
    "后处理": 0.05,
    "清理和完成": 0.03,
    "total": 1.23
  }
}
```

#### 4.4.3 语音识别 - 批量接口

```
POST /api/v1/asr
```

请求参数：
- files: 音频文件列表
- keys: 音频文件键名（逗号分隔）
- lang: 语言代码（默认"auto"）
- use_itn: 是否使用ITN（默认false）

响应示例：

```json
{
  "result": [
    {
      "key": "test1",
      "raw_text": "<|zh|><|NEUTRAL|><|Speech|>识别原始文本",
      "clean_text": "识别清洗文本",
      "text": "识别处理后文本",
      "language": "zh",
      "emotion": "NEUTRAL",
      "event": "Speech"
    },
    {
      "key": "test2",
      "raw_text": "<|en|><|HAPPY|><|Speech|>Raw text with tags",
      "clean_text": "Raw text without tags",
      "text": "Processed text",
      "language": "en",
      "emotion": "HAPPY",
      "event": "Speech"
    }
  ],
  "time_cost": 2.5,
  "detail_time": {
    "文件上传": 0.2,
    "模型推理": 2.0,
    "后处理": 0.2,
    "清理和完成": 0.1,
    "total": 2.5
  }
}
```

#### 4.4.4 流式语音识别

SenseVoice API 提供三种流式语音识别方式：

1. **HTTP流式响应**：适用于离线音频文件的流式处理
2. **WebSocket实时交互**：适用于实时语音流（如麦克风输入）
3. **兼容GitHub SenseVoice API的流式接口**

##### HTTP流式识别接口

基本流式接口 `/recognize/stream`：

```
POST /recognize/stream
```

请求参数：
- audio_file: 音频文件（表单上传）
- language: 语言代码（默认"auto"）
- use_itn: 是否使用ITN（默认true）
- chunk_size_sec: 分块大小（秒）（默认3）

或者使用Base64编码的请求：
- request_data: JSON字符串，包含以下字段：
  - audio_base64: Base64编码的音频数据
  - language: 语言代码（默认"auto"）
  - use_itn: 是否使用ITN（默认true）
  - chunk_size_sec: 分块大小（秒）（默认3）

兼容接口 `/api/v1/asr/stream`：

```
POST /api/v1/asr/stream
```

请求参数：
- files: 音频文件
- keys: 音频文件键名
- lang: 语言代码（默认"auto"）
- use_itn: 是否使用ITN（默认false）
- chunk_size_sec: 分块大小（秒）（默认3）

响应格式：返回的是新行分隔的JSON（NDJSON）格式，每行包含一个JSON对象，表示一个处理块的结果：

```json
{"success":true,"message":"部分识别结果","text":"你好","accumulated_text":"你好","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":1,"time_cost":0.345}
{"success":true,"message":"部分识别结果","text":"世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":2,"time_cost":0.326}
{"success":true,"message":"识别完成","text":"你好 世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":true,"chunk_id":2,"time_cost":0.721,"detail_time":{"流式处理开始":0.001,"处理第1块":0.345,"处理第2块":0.326,"流式处理完成":0.049}}
```

##### WebSocket流式识别接口

标准WebSocket接口 `/ws/recognize`：

1. 连接建立：
```javascript
const socket = new WebSocket('ws://localhost:8000/ws/recognize');
```

2. 认证和初始化：
```javascript
const config = {
    language: "auto",
    use_itn: true
};
socket.send(JSON.stringify(config));
```

3. 发送音频数据：
```javascript
// 发送音频块
socket.send(audioChunk);

// 发送空数据表示音频结束
socket.send(new ArrayBuffer(0));
```

4. 接收识别结果：
```javascript
socket.onmessage = function(event) {
    const result = JSON.parse(event.data);
    
    if (result.is_final) {
        // 最终结果处理
        console.log("最终识别结果:", result.text);
    } else {
        // 部分结果处理
        console.log("当前识别:", result.text);
    }
};
```

兼容接口 `/api/v1/ws/asr`：

与标准WebSocket接口类似，但配置和响应格式略有不同：

```javascript
const socket = new WebSocket('ws://localhost:8000/api/v1/ws/asr');

// 发送配置
const config = {
    lang: "auto",
    use_itn: false,
    key: "my_audio"
};
socket.send(JSON.stringify(config));
```

### 4.5 使用示例

#### Python示例

```python
import requests
import base64
import json

# 方法1：上传音频文件
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/recognize",
        files={"audio_file": f},
        data={"language": "auto", "use_itn": "true"}
    )
    print(response.json())

# 方法2：Base64编码音频
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    json_data = {
        "audio_base64": audio_base64,
        "language": "auto",
        "use_itn": True
    }
    
    response = requests.post(
        "http://localhost:8000/recognize",
        data={"request_data": json.dumps(json_data)}
    )
    print(response.json())

# 方法3：批量处理
with open("audio1.wav", "rb") as f1, open("audio2.wav", "rb") as f2:
    response = requests.post(
        "http://localhost:8000/api/v1/asr",
        files=[
            ("files", ("audio1.wav", f1.read())),
            ("files", ("audio2.wav", f2.read()))
        ],
        data={
            "keys": "test1,test2",
            "lang": "auto",
            "use_itn": "true"
        }
    )
    print(response.json())

# 方法4：HTTP流式处理
with open("audio.wav", "rb") as f:
    # 使用stream=True接收流式响应
    response = requests.post(
        "http://localhost:8000/recognize/stream",
        files={"audio_file": f},
        data={"language": "auto", "use_itn": "true", "chunk_size_sec": "3"},
        stream=True
    )
    
    # 处理流式响应
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode('utf-8'))
            print(f"当前文本: {result.get('text')}")
            print(f"累积文本: {result.get('accumulated_text')}")
            print(f"是否最终结果: {result.get('is_final', False)}")
            
# 方法5：WebSocket流式处理
import asyncio
import websockets

async def websocket_asr():
    # 连接到WebSocket服务器
    async with websockets.connect("ws://localhost:8000/ws/recognize") as ws:
        # 发送配置信息
        await ws.send(json.dumps({
            "language": "auto",
            "use_itn": True
        }))
        
        # 读取并发送音频数据
        with open("audio.wav", "rb") as f:
            while True:
                chunk = f.read(3200)  # 读取约0.2秒的音频数据（16kHz采样率）
                if not chunk:
                    break
                    
                # 发送音频块
                await ws.send(chunk)
                
                # 接收并处理结果
                result = json.loads(await ws.recv())
                print(f"当前识别: {result.get('text')}")
                
            # 发送空数据表示结束
            await ws.send(b"")
            
            # 接收最终结果
            final_result = json.loads(await ws.recv())
            print(f"最终识别结果: {final_result.get('text')}")

# 运行WebSocket示例
# asyncio.run(websocket_asr())
```

#### curl示例

```bash
# 上传单个音频文件
curl -X POST http://localhost:8000/recognize \
  -F "audio_file=@audio.wav" \
  -F "language=auto" \
  -F "use_itn=true"

# 批量处理多个音频文件
curl -X POST "http://localhost:8000/api/v1/asr" \
  -F "files=@audio1.mp3" \
  -F "files=@audio2.mp3" \
  -F "keys=test1,test2" \
  -F "lang=auto" \
  -w "总耗时: %{time_total}秒\n"

# HTTP流式处理
curl -X POST "http://localhost:8000/recognize/stream" \
  -F "audio_file=@audio.wav" \
  -F "language=auto" \
  -F "use_itn=true" \
  -F "chunk_size_sec=3"

# WebSocket流式处理需要使用WebSocket客户端，如websocket-client库或专门的WebSocket测试工具
```

### 4.6 错误处理

服务在出现错误时会返回相应的HTTP状态码和详细错误信息：

- 400 Bad Request: 请求参数错误
- 500 Internal Server Error: 服务器内部错误

错误响应示例：

```json
{
  "success": false,
  "message": "处理失败: 错误详情",
  "time_cost": 0.05,
  "detail_time": {
    "文件上传": 0.05,
    "total": 0.05
  }
}
```

## 5. 许可和贡献

SenseVoice API 使用 MIT 许可证。欢迎通过 Issues 和 Pull Requests 进行贡献和改进。 