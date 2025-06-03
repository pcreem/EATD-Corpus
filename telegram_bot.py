import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from feature import AudioTextEmotionModel, extract_audio_features, extract_text_features
import whisper

# 初始化模型
MODEL_PATH = "model_weights.pth"
EMOTION_LABELS = {0: '正面', 1: '中性', 2: '負面'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_model = AudioTextEmotionModel(audio_input_dim=180, text_input_dim=768, hidden_dim=128, output_dim=3)
emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
emotion_model.to(device)
emotion_model.eval()

whisper_model = whisper.load_model("base")

# 從環境變數中讀取 Telegram API Token
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN 未設置，請確認環境變數配置。")

# 處理 /start 命令
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！歡迎使用語音情緒辨識 Bot。請傳送一段語音或文字給我。")

# 處理文本消息
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    text_feat = extract_text_features(text)
    text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 預測情緒
    with torch.no_grad():
        output = emotion_model(torch.zeros(1, 1, 180).to(device), text_tensor)
        pred = torch.argmax(output, dim=1).item()

    await update.message.reply_text(f"我覺得你的情緒是：{EMOTION_LABELS[pred]}")

# 處理語音消息
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.voice.file_id)
    audio_path = "user_input.ogg"
    await file.download_to_drive(audio_path)

    # 語音轉文字
    result = whisper_model.transcribe(audio_path, language="zh")
    text = result["text"]

    # 特徵提取與情緒辨識
    audio_feat = extract_audio_features(audio_path)
    text_feat = extract_text_features(text)
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = emotion_model(audio_tensor, text_tensor)
        pred = torch.argmax(output, dim=1).item()

    await update.message.reply_text(f"語音轉文字結果：{text}\n預測情緒：{EMOTION_LABELS[pred]}")

# 初始化 Telegram Bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_audio))

    print("Bot 正在運行中...")
    app.run_polling()
