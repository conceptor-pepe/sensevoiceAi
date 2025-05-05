# SenseVoice API 服务

基于SenseVoice Small模型的ONNX部署API服务。

## 功能特点

- 基于ONNX Runtime优化推理速度
- 支持特定GPU设备指定
- 提供同步REST API接口
- 支持多种音频输入方式（文件上传和Base64编码）
- 返回识别文本、语言类型、情绪和事件信息

## 环境要求

- Python 3.8+
- CUDA 11.0+（GPU加速）
- 足够的内存和磁盘空间用于模型加载

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务

### 方法1：直接运行

```bash
# 赋予启动脚本执行权限
chmod +x start.sh

# 使用默认配置启动
./start.sh

# 或者自定义配置
SENSEVOICE_GPU_DEVICE=1 SENSEVOICE_PORT=8080 ./start.sh
```

### 方法2：Docker部署

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

## 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| SENSEVOICE_MODEL_DIR | 模型目录 | iic/SenseVoiceSmall |
| SENSEVOICE_GPU_DEVICE | GPU设备ID | 0 |
| SENSEVOICE_HOST | 监听地址 | 0.0.0.0 |
| SENSEVOICE_PORT | 监听端口 | 8000 |
| SENSEVOICE_BATCH_SIZE | 批处理大小 | 1 |

## API接口

### 1. 健康检查

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

### 2. 语音识别

```
POST /recognize
```

#### 支持两种请求方式：

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

#### 响应格式：

```json
{
  "success": true,
  "message": "识别成功",
  "text": "识别出的文本内容",
  "language": "zh",
  "emotion": "NEUTRAL",
  "event": "Speech",
  "time_cost": 1.23
}
```

## 使用示例

### Python请求示例

```python
import requests
import base64

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
```

### curl示例

```bash
# 上传音频文件
curl -X POST http://localhost:8000/recognize \
  -F "audio_file=@en.mp3" \
  -F "language=auto" \
  -F "use_itn=true" \
  -w "总耗时: %{time_total}秒\n" \
  -s

curl -X POST "http://localhost:8000/api/v1/asr" \
  -F "files=@en.mp3" \
  -F "keys=test" \
  -F "lang=en" \
  -w "总耗时: %{time_total}秒\n" \
  -s

# 上传多个音频文件
curl -X POST "http://localhost:8000/api/v1/asr" \
  -F "files=@en.mp3" \
  -F "files=@zh.mp3" \
  -F "keys=test1,test2" \
  -F "lang=en" \
  -w "总耗时: %{time_total}秒\n" \
  -s

# Base64编码请求
curl -X POST http://localhost:8000/recognize \
  -F 'request_data={"audio_base64":"BASE64_ENCODED_AUDIO_DATA", "language":"auto", "use_itn":true}'
```

## 注意事项

1. 请确保您有足够的GPU内存来加载模型
2. 对于长音频文件，处理时间会相应增加
3. 临时文件存储在`/tmp`目录下，请确保有足够的空间
4. 服务在处理完请求后会自动清理临时文件 