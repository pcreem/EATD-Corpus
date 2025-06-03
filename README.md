# 🎧 中文語音情緒辨識系統（基於 EATD-Corpus）

本專案整合了語音與文字特徵，實現了一個即時辨識使用者情緒的中文語音情緒辨識模型，並提供可部署於 Telegram 的互動 Bot，適合心理輔助、人機互動及智慧照護等場景。

---

## 🔍 專案背景
隨著 AI 技術進步，情緒辨識系統被廣泛應用於心理健康與智能助理中。然而，中文語音情緒資料集稀缺，成為研究瓶頸。本專案基於 **EATD-Corpus**（首個公開中文語音情緒語料庫），建構一個融合語音與文字的多模態辨識模型，並支持即時互動。

---

## 🎯 專案特色
- 📦 對語音與文字資料進行特徵提取與預處理
- 🧠 使用 GRU 和 BiLSTM 融合語音及文字通道
- 🤖 提供 Telegram Bot 介面，支援即時情緒分析
- 🌐 支援 Docker 容器化部署至 Render 平台

---

## 📁 系統架構

```text
Audio Features (180-d) ──► GRU ─► BiLSTM ─┐
                                         │
Text Features (768-d) ─────► BiLSTM ─────┘──► 拼接 ─► 全連接層 ─► Softmax（3 類別）
```
- **音訊通道**：GRU → BiLSTM
- **文字通道**：BiLSTM
- **融合層**：拼接後送入全連接層與 Softmax 分類

---

## 🧰 安裝與執行

### 1️⃣ 安裝依賴
建議使用 Python 3.8+：
```bash
pip install -r requirements.txt
```

### 2️⃣ 本地執行 Telegram Bot
1. 確保已安裝 `model_weights.pth` 至專案根目錄。
2. 設定環境變數 `TELEGRAM_TOKEN`，或直接修改程式以硬編碼 Token。
3. 執行 Bot 程式：
   ```bash
   python telegram_bot.py
   ```

---

## 🚀 部署指南

### 使用 Docker 部署
1. 確保專案包含以下文件：
    - `Dockerfile`
    - `requirements.txt`
    - `telegram_bot.py`
    - `model_weights.pth`
    - `start.sh`

2. Dockerfile 範例：
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    COPY . /app
    RUN pip install -r requirements.txt
    CMD ["bash", "start.sh"]
    ```

3. 啟動腳本 `start.sh` 範例：
    ```bash
    #!/bin/bash
    export TELEGRAM_TOKEN=<你的 Telegram Token>
    python telegram_bot.py
    ```

4. 執行以下命令構建並啟動容器：
    ```bash
    docker build -t telegram-emotion-bot .
    docker run -e TELEGRAM_TOKEN=<你的 Telegram Token> -p 5000:5000 telegram-emotion-bot
    ```

### 在 Render 平台部署
1. 登錄 [Render](https://render.com/) 並創建 Web Service。
2. 指定代碼倉庫並設置環境變數（如 `TELEGRAM_TOKEN`）。
3. 選擇 Docker 配置並部署。

---

## 📋 目錄結構

```
.
├── telegram_bot.py         # Telegram Bot 入口程式
├── model_weights.pth      # 已訓練模型權重
├── requirements.txt       # 相依套件
├── Dockerfile             # 容器配置文件
├── start.sh               # 啟動腳本
├── README.md              # 專案說明文件
```

---

## 📌 結語

本專案展示了中文語音情緒辨識的完整流程，並成功整合至即時互動系統（Telegram Bot）。可進一步應用於偏鄉心理照護、客服助理及智慧裝置交互。
