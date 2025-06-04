#!/bin/bash
set -e

# 確保下載模型權重
python download_weights.py

# 啟動 Telegram Bot
python telegram_bot.py
