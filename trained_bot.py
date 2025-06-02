#語音特徵提取
import librosa
import numpy as np

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    features = np.concatenate((np.mean(mfcc, axis=1),
                               np.mean(chroma, axis=1),
                               np.mean(spec, axis=1)))
    return features

#文字特徵處理

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().detach().numpy()

#模型建構與訓練
import torch.nn as nn

class AudioTextEmotionModel(nn.Module):
    def __init__(self, audio_input_dim, text_input_dim, hidden_dim, output_dim):
        super(AudioTextEmotionModel, self).__init__()
        self.audio_gru = nn.GRU(audio_input_dim, hidden_dim, batch_first=True)
        self.audio_bilstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_bilstm = nn.LSTM(text_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, audio_input, text_input):
        audio_out, _ = self.audio_gru(audio_input)
        audio_out, _ = self.audio_bilstm(audio_out)
        text_out, _ = self.text_bilstm(text_input)
        combined = torch.cat((audio_out[:, -1, :], text_out[:, -1, :]), dim=1)
        output = self.fc(combined)
        return self.softmax(output)

#終端機互動介面
def main():
    print("歡迎使用中文語音情緒辨識系統！")
    audio_path = input("請輸入語音檔案路徑：")
    text_input = input("請輸入對應的文字內容：")

    audio_features = extract_audio_features(audio_path)
    text_features = extract_text_features(text_input)

    # 將特徵轉換為張量
    audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    text_tensor = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 初始化模型（請根據實際情況設定參數）
    model = AudioTextEmotionModel(audio_input_dim=audio_tensor.size(2),
                                  text_input_dim=text_tensor.size(2),
                                  hidden_dim=128,
                                  output_dim=3)  # 假設有三種情緒類別

    # 載入訓練好的模型權重
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    # 進行預測
    with torch.no_grad():
        prediction = model(audio_tensor, text_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()

    emotion_labels = {0: '正面', 1: '中性', 2: '負面'}
    print(f"預測的情緒為：{emotion_labels[predicted_class]}")

if __name__ == "__main__":
    main()

