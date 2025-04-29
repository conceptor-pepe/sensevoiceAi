# SenseVoice API 客户端使用说明

本文档详细介绍了SenseVoice API示例客户端的使用方法。

## 安装依赖

在使用示例客户端前，请确保安装必要的依赖：

```bash
pip install requests argparse
```

## 基本用法

示例客户端提供了命令行界面，使您可以轻松地与SenseVoice API交互。

基本用法格式：

```bash
python example_client.py --action <操作> [参数]
```

### 可用操作

| 操作 | 描述 |
|------|------|
| health | 检查API服务的健康状态 |
| models | 获取可用的语音识别模型信息 |
| transcribe | 转录单个音频文件 |
| batch | 批量转录多个音频文件 |
| srt | 转录音频并保存为SRT字幕文件 |
| performance | 获取API性能统计信息 |
| system | 获取系统资源指标 |
| cache | 获取缓存统计信息 |

### 通用参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --url | API服务的URL | http://localhost:8000 |
| --api-key | API密钥（如启用了认证） | 无 |
| --language | 音频语言代码 | auto |

## 使用示例

### 1. 检查API健康状态

```bash
python example_client.py --action health
```

### 2. 获取可用模型信息

```bash
python example_client.py --action models
```

### 3. 转录单个音频文件

```bash
python example_client.py --action transcribe --file path/to/audio.wav --language zh
```

### 4. 批量转录多个音频文件

```bash
python example_client.py --action batch --files path/to/audio1.wav path/to/audio2.wav --language zh
```

### 5. 转录音频并保存为SRT字幕文件

```bash
python example_client.py --action srt --file path/to/audio.wav --output path/to/subtitle.srt --language zh
```

### 6. 获取性能统计信息

```bash
python example_client.py --action performance
```

### 7. 获取系统资源指标

```bash
python example_client.py --action system
```

### 8. 获取缓存统计信息

```bash
python example_client.py --action cache
```

## 在代码中使用客户端类

您也可以在自己的Python代码中导入并使用`SenseVoiceClient`类：

```python
from example_client import SenseVoiceClient

# 创建客户端实例
client = SenseVoiceClient(base_url="http://localhost:8000", api_key="your-api-key")

# 转录音频文件
result = client.transcribe_audio("path/to/audio.wav", language="zh")
print(result)

# 批量转录
results = client.batch_transcribe(["file1.wav", "file2.wav"], language="auto")

# 创建SRT字幕
client.transcribe_to_srt("video.wav", "video.srt", language="en")
```

## 支持的语言

SenseVoice API支持以下语言：

- `auto`: 自动检测语言
- `zh`: 中文(普通话)
- `en`: 英语
- `yue`: 粤语
- `ja`: 日语
- `ko`: 韩语

## 响应格式

转录接口支持多种响应格式：

- `json`: JSON格式（完整的转录结果）
- `text`: 纯文本格式
- `srt`: SRT字幕格式
- `vtt`: WebVTT字幕格式

## 错误处理

示例客户端提供了基本的错误处理。如果API请求失败，将显示相应的错误信息。

## 高级用法

### 自定义API服务地址

如果您的API服务不是运行在默认地址，可以使用`--url`参数指定：

```bash
python example_client.py --url http://192.168.1.100:9000 --action health
```

### 使用API密钥

如果您的API服务启用了认证，可以使用`--api-key`参数提供密钥：

```bash
python example_client.py --api-key your-api-key-here --action transcribe --file audio.wav
``` 