import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import numpy as np
import librosa
from transformers import BertTokenizer, BertModel
from your_model import AudioTextEmotionModel, extract_audio_features, extract_text_features  # 用你的模組

# === 參數設定 ===
SAMPLE_RATE = 16000
DURATION = 5  # 錄音秒數
MODEL_PATH = "model_weights.pth"
EMOTION_LABELS = {0: '正面', 1: '中性', 2: '負面'}

# === 載入模型 ===
emotion_model = AudioTextEmotionModel(audio_input_dim=180, text_input_dim=768, hidden_dim=128, output_dim=3)
emotion_model.load_state_dict(torch.load(MODEL_PATH))
emotion_model.eval()

# === Whisper 模型載入 ===
whisper_model = whisper.load_model("base")

# === BERT tokenizer/model ===
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

# === 錄音函數 ===
def record_audio(filename="output.wav"):
    print(f"🎤 請開始說話（{DURATION} 秒）...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("✅ 錄音完成！")

# === 預測流程 ===
def predict_emotion(audio_path):
    # 語音轉文字
    result = whisper_model.transcribe(audio_path, language="zh")
    text = result["text"]
    print(f"📝 語音轉文字結果：{text}")

    # 特徵處理
    audio_feat = extract_audio_features(audio_path)
    text_feat = extract_text_features(text)

    # 張量轉換
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 預測
    with torch.no_grad():
        output = emotion_model(audio_tensor, text_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"🔍 預測情緒為：{EMOTION_LABELS[pred]}")
    return text, EMOTION_LABELS[pred]

# === 主執行 ===
if __name__ == "__main__":
    while True:
        input("🔘 按下 Enter 鍵開始錄音（或 Ctrl+C 離開）")
        record_audio("user_input.wav")
        text, emotion = predict_emotion("user_input.wav")
        print(f"🤖 ChatBot 回應：你說的是「{text}」，我覺得你現在的情緒是「{emotion}」。")
