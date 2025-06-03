# your_model.py

import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import BertTokenizer, BertModel

# === 模型結構 ===
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

# === 音訊特徵萃取 ===
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    features = np.concatenate((
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec, axis=1)
    ))
    return features

# === 文字特徵萃取（使用 BERT） ===
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().detach().numpy()
