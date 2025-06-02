import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# 設定基本參數
base_path = r"C:\Users\ittraining\Downloads\EATD-Corpus\EATD-Corpus"
emotion_labels = {'positive': 0, 'neutral': 1, 'negative': 2}
batch_size = 16
num_epochs = 10
hidden_dim = 128
output_dim = 3

# 初始化 BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 萃取音訊特徵
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    features = np.concatenate((np.mean(mfcc, axis=1),
                               np.mean(chroma, axis=1),
                               np.mean(spec, axis=1)))
    return features

# 萃取文字特徵
def extract_text_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().detach().numpy()

# 自訂 Dataset 類別
class EmotionDataset(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['audio_path']
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        audio_feat = extract_audio_features(audio_path)
        text_feat = extract_text_features(text)
        return (torch.tensor(audio_feat, dtype=torch.float32),
                torch.tensor(text_feat, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))

# 定義模型
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

# 建立 DataFrame
def build_dataframe():
    rows = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for emotion in emotion_labels:
                audio_file = os.path.join(folder_path, f"{emotion}.wav")
                text_file = os.path.join(folder_path, f"{emotion}.txt")
                if os.path.exists(audio_file) and os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    rows.append({
                        'audio_path': audio_file,
                        'text': text,
                        'label': emotion_labels[emotion]
                    })
    return pd.DataFrame(rows)

# 主流程
def main():
    print("🔍 建立資料集 DataFrame...")
    df = build_dataframe()
    print(f"總樣本數: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = EmotionDataset(train_df)
    val_dataset = EmotionDataset(val_df)

    # 提取一筆範例特徵確認維度
    audio_feat_dim = len(extract_audio_features(train_df.iloc[0]['audio_path']))
    text_feat_dim = extract_text_features(train_df.iloc[0]['text']).shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AudioTextEmotionModel(audio_feat_dim, text_feat_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🚀 開始訓練模型...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for audio_feat, text_feat, labels in train_loader:
            audio_feat = audio_feat.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
            outputs = model(audio_feat, text_feat)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 驗證階段
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for audio_feat, text_feat, labels in val_loader:
                audio_feat = audio_feat.unsqueeze(1)
                text_feat = text_feat.unsqueeze(1)
                outputs = model(audio_feat, text_feat)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f} | Val Acc: {val_acc:.2%}")

    print("✅ 模型訓練完成，儲存權重至 model_weights.pth")
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    main()
