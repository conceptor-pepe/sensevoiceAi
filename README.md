# SenseVoice-python

基于FunASR的中文/英文语音识别API服务，使用SenseVoiceSmall模型。

## 功能特点

- 支持中文和英文的语音识别转写
- 基于FastAPI的高性能异步处理
- 支持批量音频文件上传和处理
- 大文件自动分片处理功能
- 内存监控和资源管理
- 并发请求控制和限流
- 健康检查端点
- 详细的日志记录

## 新增功能

### 大文件处理

系统现在支持自动处理大型音频文件：
- 自动检测大文件（基于文件大小或音频时长）
- 大文件自动分片处理，然后合并结果
- 可配置的分片大小
- 支持超长音频文件（小时级别）的处理

### 并发控制和资源管理

- 请求信号量机制限制最大并发请求数
- 内存监控，防止内存溢出
- 系统资源不足时自动拒绝新请求
- 实时监控并记录服务状态

## 安装

1. 克隆仓库
```
git clone https://github.com/your-repo/SenseVoice-python.git
cd SenseVoice-python
```

2. 安装依赖
```
pip install -r requirements.txt
```

3. 下载模型（如果需要）
```
# 下载模型到 iic/SenseVoiceSmall 目录
```

## 配置

编辑 `config.py` 文件以设置服务参数：
- 服务器配置（主机、端口）
- GPU设备和线程数
- 模型配置
- 并发和内存管理配置
- 日志配置

可以创建 `local_config.py` 文件来覆盖默认配置。

## 启动服务

```
python api.py
```

服务启动后，可以通过以下URL访问API文档：
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

## API使用示例

### 音频转写

```python
import requests
import os

# 服务URL
url = 'http://localhost:8000/transcribe'

# 音频文件
files = [
    ('files', ('audio1.wav', open('path/to/audio1.wav', 'rb'))),
    ('files', ('audio2.wav', open('path/to/audio2.wav', 'rb')))
]

# 文件名列表，必须与文件数量一致
data = {
    'keys': 'audio1,audio2',
    'lang': 'auto'  # 可选: 'auto', 'zh', 'en'
}

# 发送请求
response = requests.post(url, files=files, data=data)
result = response.json()

# 打印结果
print(f"状态: {result['message']}")
print(f"处理时间: {result['processing_time']} 秒")
for i, text in enumerate(result['results']):
    print(f"文件 {i+1} 转写结果: {text}")
```

### 健康检查

```python
import requests

response = requests.get('http://localhost:8000/health')
health_info = response.json()
print(f"服务状态: {health_info['status']}")
print(f"内存使用率: {health_info['memory']['usage']}")
```

## 系统要求

- Python 3.8+
- CUDA支持（如果使用GPU）
- 足够的内存（至少8GB）

## 许可证

[MIT License](LICENSE) 