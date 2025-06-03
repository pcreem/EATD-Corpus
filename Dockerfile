# 使用 Python 3.10 基礎映像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 複製專案文件到容器
COPY . /app

# 更新 apt 並安裝系統依賴
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 暴露應用程序使用的端口（Render 默認可用 8000）
EXPOSE 8000

# 設置啟動命令
CMD ["python", "telegram_bot.py"]
