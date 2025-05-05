FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY api.py .

# 设置环境变量（可以在运行时覆盖）
ENV SENSEVOICE_MODEL_DIR="iic/SenseVoiceSmall"
ENV SENSEVOICE_GPU_DEVICE="0"
ENV SENSEVOICE_HOST="0.0.0.0"
ENV SENSEVOICE_PORT="8000"
ENV SENSEVOICE_BATCH_SIZE="1"

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python", "api.py"] 