# SenseVoice API

SenseVoice API是一个高性能的语音识别API服务，基于深度学习模型，支持多语言识别、情感识别和声音事件识别。本服务提供标准API、流式识别和WebSocket实时交互，适用于各种语音识别场景。

## 功能特点

- **多语言支持**：自动检测并识别中文(zh)、英文(en)、粤语(yue)、日语(ja)、韩语(ko)
- **情感识别**：可识别说话者的情感状态，包括高兴(HAPPY)、悲伤(SAD)、愤怒(ANGRY)、恐惧(FEARFUL)、厌恶(DISGUSTED)、惊讶(SURPRISED)和中性(NEUTRAL)
- **事件识别**：能够识别背景音乐(BGM)、掌声(Applause)、笑声(Laughter)、哭声(Cry)、打喷嚏(Sneeze)、呼吸声(Breath)和咳嗽声(Cough)
- **多种接口模式**：
  - 标准API：适用于非实时场景
  - HTTP流式响应：适用于离线音频文件的流式处理
  - WebSocket实时交互：适用于实时语音流（如麦克风输入）
- **兼容性**：提供兼容GitHub SenseVoice API的接口，便于迁移现有应用

## 安装说明

### 环境要求

- Python 3.7+
- CUDA 11.0+（GPU加速，可选）
- 足够的内存和磁盘空间

### 依赖安装

```bash
# 安装必要的依赖
pip install -r requirements.txt
```

### 运行服务

使用提供的启动脚本：

```bash
# 前台运行
bash start.sh

# 后台运行（守护进程模式）
bash start.sh -d
```

### 环境变量配置

可以通过环境变量自定义服务配置：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| SENSEVOICE_HOST | 服务监听地址 | 0.0.0.0 |
| SENSEVOICE_PORT | 服务端口 | 8000 |
| SENSEVOICE_MODEL_DIR | 模型目录 | iic/SenseVoiceSmall |
| SENSEVOICE_GPU_DEVICE | GPU设备ID | 自动检测 |
| SENSEVOICE_LOG_LEVEL | 日志级别 | INFO |
| SENSEVOICE_TEMP_DIR | 临时文件目录 | /tmp/sensevoice |

## API文档

### 1. 标准API接口

#### 1.1 `/recognize` - 标准语音识别

**请求方法**：POST

**参数**：
- `audio_file`：音频文件（FormData）
- `language`：语言代码，可选值为 `auto`、`zh`、`en` 等，默认为 `auto`
- `use_itn`：是否使用反向文本归一化，默认为 `true`

**或者**：
- `request_data`：包含 `audio_base64`、`language`、`use_itn` 的JSON字符串

**响应示例**：
```json
{
  "success": true,
  "message": "识别成功",
  "text": "你好世界",
  "clean_text": "你好世界",
  "raw_text": "<|zh|><|NEUTRAL|><|Speech|>你好世界",
  "language": "zh",
  "emotion": "NEUTRAL",
  "event": "Speech",
  "time_cost": 0.325
}
```

#### 1.2 `/api/v1/asr` - 兼容接口

**请求方法**：POST

**参数**：
- `files`：音频文件列表（FormData）
- `keys`：与音频文件对应的键名列表，多个用逗号分隔，默认为 `audio`
- `lang`：语言代码，可选值为 `auto`、`zh`、`en` 等，默认为 `auto`
- `use_itn`：是否使用反向文本归一化，默认为 `true`

**响应示例**：
```json
{
  "result": [
    {
      "key": "audio1",
      "raw_text": "<|zh|><|NEUTRAL|><|Speech|>你好世界",
      "clean_text": "你好世界",
      "text": "你好世界",
      "language": "zh",
      "emotion": "NEUTRAL",
      "event": "Speech"
    }
  ],
  "time_cost": 0.358
}
```

### 2. HTTP流式识别接口

#### 2.1 `/recognize/stream` - 流式识别

**请求方法**：POST

**参数**：
- `audio_file`：音频文件（FormData）
- `language`：语言代码，默认为 `auto`
- `use_itn`：是否使用反向文本归一化，默认为 `true`
- `chunk_size_sec`：分块大小（秒），默认为 `3`

**或者**：
- `request_data`：包含 `audio_base64`、`language`、`use_itn`、`chunk_size_sec` 的JSON字符串

**响应**：新行分隔的JSON（NDJSON）格式，每行包含一个JSON对象，表示一个处理块的结果

```json
{"success":true,"message":"部分识别结果","text":"你好","accumulated_text":"你好","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":1,"time_cost":0.345}
{"success":true,"message":"部分识别结果","text":"世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":2,"time_cost":0.326}
{"success":true,"message":"识别完成","text":"你好 世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":true,"chunk_id":2,"time_cost":0.721,"detail_time":{"流式处理开始":0.001,"处理第1块":0.345,"处理第2块":0.326,"流式处理完成":0.049}}
```

#### 2.2 `/api/v1/asr/stream` - 兼容流式接口

**请求方法**：POST

**参数**：
- `files`：音频文件（FormData）
- `keys`：与音频文件对应的键名，默认为 `audio`
- `lang`：语言代码，默认为 `auto`
- `use_itn`：是否使用反向文本归一化，默认为 `true`
- `chunk_size_sec`：分块大小（秒），默认为 `3`

**响应**：与 `/recognize/stream` 相同，为新行分隔的JSON流

### 3. WebSocket接口

#### 3.1 `/ws/recognize` - WebSocket接口

**连接URL**：`ws://<host>:<port>/ws/recognize`

**初始消息**（JSON格式）：
```json
{
  "language": "auto",
  "use_itn": true
}
```

**音频数据**：连接建立后，客户端发送二进制音频数据

**空数据**：发送空的二进制数据表示音频结束

**响应示例**：
```json
{
  "success": true,
  "message": "部分识别结果",
  "text": "你好",
  "accumulated_text": "你好",
  "language": "zh",
  "emotion": "NEUTRAL",
  "event": "Speech",
  "is_final": false,
  "chunk_id": 1,
  "time_cost": 0.345
}
```

#### 3.2 `/api/v1/ws/asr` - 兼容WebSocket接口

**连接URL**：`ws://<host>:<port>/api/v1/ws/asr`

**初始消息**（JSON格式）：
```json
{
  "lang": "auto",
  "use_itn": true,
  "key": "audio1"
}
```

**响应示例**：
```json
{
  "result": [
    {
      "key": "audio1",
      "text": "你好世界",
      "language": "zh",
      "emotion": "NEUTRAL"
    }
  ],
  "is_final": false
}
```

### 4. 健康检查接口

#### 4.1 `/health` - 服务健康状态

**请求方法**：GET

**响应示例**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_dir": "iic/SenseVoiceSmall",
  "gpu_device": "0"
}
```

## 客户端示例

项目包含两个示例客户端：

### Python客户端

文件：`websocket_client_example.py`

使用方法：
```bash
python websocket_client_example.py test.wav zh true
```

### Web客户端

文件：`websocket_client_example.html`

直接在浏览器中打开此HTML文件，可以通过麦克风录音或上传音频文件进行流式识别演示。

## 性能优化

1. **GPU加速**：服务默认使用GPU加速（如果可用），可以通过环境变量指定GPU设备
2. **批处理**：可以通过设置 `SENSEVOICE_BATCH_SIZE` 环境变量来调整批处理大小
3. **临时文件**：所有临时文件会自动清理，可以通过 `SENSEVOICE_TEMP_DIR` 指定临时目录

## 故障排除

### 常见问题

1. **模型加载失败**：
   - 检查模型目录是否正确
   - 确保有足够的内存和磁盘空间
   - 查看日志以获取详细错误信息

2. **GPU不可用**：
   - 检查NVIDIA驱动是否正确安装
   - 确认CUDA环境配置正确
   - 尝试使用CPU模式（设置 `SENSEVOICE_GPU_DEVICE=-1`）

3. **性能问题**：
   - 调整分块大小（流式处理）
   - 优化音频格式和采样率
   - 考虑使用更小的模型 