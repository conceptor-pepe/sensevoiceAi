# SenseVoice API 文档

## 概述

SenseVoice API 是一个高性能的语音识别服务，提供多语言音频转录、语音识别和字幕生成功能。本文档详细介绍了API的各个端点、参数以及使用方法。

## 基础信息

- **基础URL**: `http://<host>:<port>`
- **默认端口**: 8000
- **API版本**: v1
- **认证方式**: API密钥 (可选，通过 `X-API-Key` 请求头传递)

## 健康检查

### 获取API健康状态

```
GET /api/v1/health
```

返回API服务的健康状态信息。

#### 响应示例:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime": "1h 23m 45s"
}
```

## 模型管理

### 获取可用模型列表

```
GET /api/v1/models
```

返回所有可用的转录模型信息。

#### 响应示例:

```json
{
  "models": [
    {
      "id": "sense-voice-small",
      "name": "SenseVoice Small",
      "description": "轻量级通用语音识别模型",
      "languages": ["zh", "en", "yue", "ja", "ko"],
      "size": "小型",
      "default": true
    },
    {
      "id": "sense-voice-medium",
      "name": "SenseVoice Medium",
      "description": "中等规模的语音识别模型，平衡速度和准确性",
      "languages": ["zh", "en", "yue", "ja", "ko"],
      "size": "中型",
      "default": false
    },
    {
      "id": "sense-voice-large",
      "name": "SenseVoice Large",
      "description": "大型高精度语音识别模型",
      "languages": ["zh", "en", "yue", "ja", "ko"],
      "size": "大型",
      "default": false
    }
  ]
}
```

## 音频转录

### 转录音频文件

```
POST /api/v1/audio/transcriptions
```

将上传的音频文件转录为文本。

#### 请求参数:

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| file | 文件 | 是 | 要转录的音频文件 |
| model | 字符串 | 否 | 使用的模型ID (默认: "sense-voice-small") |
| language | 字符串 | 否 | 音频的语言代码 (默认: "auto") |
| response_format | 字符串 | 否 | 响应格式 (json, text, srt, vtt) (默认: "json") |
| timestamps | 布尔值 | 否 | 是否返回单词级时间戳 (默认: false) |
| word_timestamps | 布尔值 | 否 | 是否返回单词级时间戳 (默认: false) |
| task | 字符串 | 否 | 任务类型 (transcribe, translate) (默认: "transcribe") |
| prompt | 字符串 | 否 | 可选提示以引导转录 |

#### 响应示例 (JSON):

```json
{
  "text": "这是转录的文本内容",
  "language": "zh",
  "duration": 15.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "这是第一段"
    },
    {
      "id": 1,
      "start": 3.5,
      "end": 7.2,
      "text": "这是第二段"
    },
    {
      "id": 2,
      "start": 7.8,
      "end": 15.5,
      "text": "这是最后一段"
    }
  ],
  "model_id": "sense-voice-small"
}
```

#### 响应示例 (SRT):

```
1
00:00:00,000 --> 00:00:03,500
这是第一段

2
00:00:03,500 --> 00:00:07,200
这是第二段

3
00:00:07,800 --> 00:00:15,500
这是最后一段
```

### 批量转录音频文件

```
POST /api/v1/audio/batch_transcriptions
```

批量转录多个音频文件。

#### 请求参数:

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| files | 文件数组 | 是 | 要转录的音频文件数组 |
| model | 字符串 | 否 | 使用的模型ID (默认: "sense-voice-small") |
| language | 字符串 | 否 | 音频的语言代码 (默认: "auto") |
| response_format | 字符串 | 否 | 响应格式 (json, text) (默认: "json") |
| task | 字符串 | 否 | 任务类型 (transcribe, translate) (默认: "transcribe") |

#### 响应示例:

```json
{
  "results": [
    {
      "filename": "audio1.mp3",
      "text": "第一个音频文件的转录内容",
      "language": "zh",
      "duration": 12.3
    },
    {
      "filename": "audio2.mp3",
      "text": "第二个音频文件的转录内容",
      "language": "en",
      "duration": 8.7
    }
  ],
  "total_files": 2,
  "total_duration": 21.0,
  "model_id": "sense-voice-small"
}
```

## 性能监控

### 获取性能统计信息

```
GET /api/v1/performance
```

获取API的性能统计信息。

#### 响应示例:

```json
{
  "function_stats": {
    "transcribe_audio": {
      "count": 150,
      "avg_time": 2.3,
      "min_time": 0.8,
      "max_time": 5.6,
      "p95_time": 4.2,
      "total_time": 345.0
    },
    "batch_transcribe": {
      "count": 25,
      "avg_time": 8.7,
      "min_time": 3.2,
      "max_time": 18.5,
      "p95_time": 15.3,
      "total_time": 217.5
    }
  },
  "api_stats": {
    "total_requests": 1250,
    "success_rate": 98.5,
    "avg_response_time": 1.8
  }
}
```

### 获取性能摘要

```
GET /api/v1/performance/summary
```

获取API性能的简要摘要信息。

#### 响应示例:

```json
{
  "total_requests": 1250,
  "avg_response_time": 1.8,
  "success_rate": 98.5,
  "top_functions": [
    {
      "name": "transcribe_audio",
      "count": 150,
      "avg_time": 2.3
    },
    {
      "name": "batch_transcribe",
      "count": 25,
      "avg_time": 8.7
    }
  ],
  "recent_status": "healthy"
}
```

## 系统监控

### 获取系统指标

```
GET /api/v1/system/metrics
```

获取服务器的系统资源使用情况。

#### 响应示例:

```json
{
  "cpu": {
    "usage_percent": 45.2,
    "core_count": 8,
    "load_avg": [2.5, 2.3, 2.1]
  },
  "memory": {
    "total": 16384,
    "used": 8192,
    "free": 8192,
    "usage_percent": 50.0
  },
  "disk": {
    "total": 512000,
    "used": 128000,
    "free": 384000,
    "usage_percent": 25.0
  },
  "gpu": {
    "available": true,
    "count": 2,
    "devices": [
      {
        "name": "NVIDIA GeForce RTX 3080",
        "memory_total": 10240,
        "memory_used": 3072,
        "usage_percent": 30.0,
        "temperature": 65
      },
      {
        "name": "NVIDIA GeForce RTX 3080",
        "memory_total": 10240,
        "memory_used": 2048,
        "usage_percent": 20.0,
        "temperature": 60
      }
    ]
  },
  "timestamp": "2023-08-15T13:24:56Z"
}
```

### 获取系统摘要

```
GET /api/v1/system/summary
```

获取系统资源使用的简要摘要。

#### 响应示例:

```json
{
  "status": "healthy",
  "cpu_usage": 45.2,
  "memory_usage": 50.0,
  "disk_usage": 25.0,
  "gpu_usage": 25.0,
  "uptime": "3d 12h 45m",
  "active_workers": 4,
  "timestamp": "2023-08-15T13:24:56Z"
}
```

## 缓存管理

### 获取缓存统计

```
GET /api/v1/cache/stats
```

获取API缓存使用的统计信息。

#### 响应示例:

```json
{
  "hits": 875,
  "misses": 375,
  "hit_rate": 70.0,
  "size": 1024,
  "max_size": 2048,
  "usage_percent": 50.0,
  "avg_lookup_time": 0.002,
  "avg_miss_penalty": 2.5,
  "items_count": 250
}
```

### 清除缓存

```
POST /api/v1/cache/clear
```

清除API的缓存数据。

#### 响应示例:

```json
{
  "success": true,
  "cleared_items": 250,
  "freed_memory": 1024,
  "timestamp": "2023-08-15T13:30:00Z"
}
```

## 支持的语言

API支持以下语言的自动检测和转录:

| 语言代码 | 语言名称 |
|---------|---------|
| auto | 自动检测 |
| zh | 中文 (普通话) |
| en | 英语 |
| yue | 粤语 |
| ja | 日语 |
| ko | 韩语 |

## 响应格式

API支持以下响应格式:

| 格式代码 | 描述 |
|---------|------|
| json | JSON格式 (默认)，包含完整的转录信息，包括段落、时间戳等 |
| text | 纯文本格式，仅包含转录文本 |
| srt | SRT字幕格式，适用于视频应用 |
| vtt | WebVTT字幕格式，适用于Web视频应用 |

## 错误处理

API使用标准HTTP状态码指示请求的成功或失败:

| 状态码 | 描述 |
|-------|------|
| 200 | 成功 |
| 400 | 错误请求 - 请求参数无效 |
| 401 | 未授权 - 需要有效的API密钥 |
| 404 | 未找到 - 请求的资源不存在 |
| 413 | 请求实体过大 - 上传的文件太大 |
| 415 | 不支持的媒体类型 - 不支持的音频格式 |
| 429 | 请求过多 - 超过API速率限制 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 - 服务器暂时过载或维护中 |

错误响应示例:

```json
{
  "error": {
    "code": "invalid_file_format",
    "message": "不支持的音频格式，请上传MP3、WAV、FLAC或M4A格式的文件",
    "status": 415
  }
}
```

## 速率限制

API实施速率限制以确保服务的稳定性。限制信息将在响应头中返回:

- `X-RateLimit-Limit`: 在一个时间窗口内允许的最大请求数
- `X-RateLimit-Remaining`: 当前时间窗口内剩余的请求数
- `X-RateLimit-Reset`: 当前时间窗口重置的时间戳

如果超过速率限制，API将返回429状态码。

## 最佳实践

1. **使用适当的模型**: 根据您的用例选择合适的模型大小。小型模型速度更快但准确性较低，大型模型准确性更高但速度较慢。

2. **指定语言**: 尽可能指定音频的语言，这可以提高转录准确性并减少处理时间。

3. **优化音频质量**: 提供高质量的音频文件以获得更好的转录结果。减少背景噪音并确保清晰的语音。

4. **实施缓存**: 如果频繁转录相同的音频，请使用客户端缓存来减少API调用。

5. **批量处理**: 对于多个小文件，使用批量转录API以减少请求次数。

6. **监控使用情况**: 使用性能和系统监控端点来跟踪API使用情况和潜在瓶颈。

## SDK和客户端

SenseVoice API提供以下客户端实现:

- [Python客户端](example_client.py) - 用于脚本和后端集成的Python客户端
- [JavaScript客户端](js_client.html) - 用于Web应用的浏览器JavaScript客户端

## 示例

### cURL示例

转录音频文件:

```bash
curl -X POST http://localhost:8000/api/v1/audio/transcriptions \
  -H "X-API-Key: your_api_key" \
  -F "file=@audio.mp3" \
  -F "model=sense-voice-small" \
  -F "language=zh" \
  -F "response_format=json"
```

获取性能统计:

```bash
curl -X GET http://localhost:8000/api/v1/performance \
  -H "X-API-Key: your_api_key"
```

### Python示例

```python
import requests

# 转录音频
def transcribe_audio(file_path, api_url="http://localhost:8000", api_key=None, language="auto"):
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "model": "sense-voice-small",
            "language": language,
            "response_format": "json"
        }
        
        response = requests.post(
            f"{api_url}/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data
        )
        
        return response.json()

# 使用示例
result = transcribe_audio("audio.mp3", api_key="your_api_key", language="zh")
print(result["text"])
```

### JavaScript示例

```javascript
// 转录音频
async function transcribeAudio(audioFile, apiUrl = "http://localhost:8000", apiKey = null, language = "auto") {
    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("model", "sense-voice-small");
    formData.append("language", language);
    formData.append("response_format", "json");
    
    const headers = {};
    if (apiKey) {
        headers["X-API-Key"] = apiKey;
    }
    
    const response = await fetch(`${apiUrl}/api/v1/audio/transcriptions`, {
        method: "POST",
        headers: headers,
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`请求失败: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
}

// 使用示例
const fileInput = document.getElementById("audioFileInput");
const transcribeButton = document.getElementById("transcribeButton");

transcribeButton.addEventListener("click", async () => {
    if (fileInput.files.length > 0) {
        try {
            const result = await transcribeAudio(
                fileInput.files[0],
                "http://localhost:8000",
                "your_api_key",
                "zh"
            );
            console.log(result.text);
        } catch (error) {
            console.error("转录失败:", error);
        }
    }
});
```

## 联系与支持

如有问题或需要帮助，请联系我们的技术支持团队:
- 电子邮件: support@sensevoice.ai
- 文档: https://sensevoice.ai/docs
- GitHub: https://github.com/sensevoice/api 