import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# === 參數設定 ===
SAMPLE_RATE = 16000
DURATION = 5  # 錄音秒數

# === Whisper 模型載入 ===
whisper_model = whisper.load_model("base")

# === 錄音函數 ===
def record_audio(filename="output.wav"):
    print(f"🎤 請開始說話（{DURATION} 秒）...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("✅ 錄音完成！")

# === 語音轉文字函數 ===
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="zh")
    text = result["text"]
    print(f"📝 語音轉文字結果：{text}")
    return text

# === 主執行 ===
if __name__ == "__main__":
    while True:
        input("🔘 按下 Enter 鍵開始錄音（或 Ctrl+C 離開）")
        record_audio("user_input.wav")
        text = transcribe_audio("user_input.wav")
        print(f"🤖 ChatBot 回應：你說的是「{text}」。")
