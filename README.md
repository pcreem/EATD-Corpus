# 🎧 中文語音情緒辨識系統（基於 EATD-Corpus）

本專案結合語音與文字特徵，實作一套能即時辨識使用者情緒的中文語音情緒辨識模型，應用於心理輔助、人機互動與智慧照護等場景。

---

## 🔍 專案背景

隨著 AI 技術進步，情緒辨識系統被廣泛應用於心理健康與智能助理中。然而，中文語音情緒資料集稀缺，成為研究瓶頸。本專案基於 **EATD-Corpus**（首個公開中文語音情緒語料庫），建構一個融合語音與文字的多模態辨識模型。

---

## 🎯 專案目標

- 📦 對語音與文字資料進行預處理與特徵萃取
- 🧠 建立融合 GRU / BiLSTM 架構的情緒分類模型
- 🧪 評估模型性能與應用於終端互動系統
- 🌐 可進一步整合至 LINE Bot 或行動裝置應用

---

## 🧱 系統架構與方法

### 📂 資料集：EATD-Corpus
- 162 位受測者錄製語音
- 每筆資料包含 `.wav` 與對應 `.txt`
- 分為「正面」、「中性」、「負面」三類情緒

### 🔍 特徵提取

#### 🎵 語音特徵（維度共 180）
- **MFCC**（40 維）
- **Chroma**（12 維）
- **Mel Spectrogram**（128 維）

```python
mfcc = librosa.feature.mfcc(...)
chroma = librosa.feature.chroma_stft(...)
spec = librosa.feature.melspectrogram(...)
features = np.concatenate([mean(mfcc), mean(chroma), mean(spec)])
```

#### 📝 文字特徵
- 使用 BERT (`bert-base-chinese`)
- 取得 `[CLS]` 向量作為 768 維特徵

```python
inputs = tokenizer(text, return_tensors='pt')
outputs = bert_model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]
```

---

## 🧠 模型架構

```text
Audio Features (180-d) ──► GRU ─► BiLSTM ─┐
                                         │
Text Features (768-d) ─────► BiLSTM ─────┘──► 拼接 ─► 全連接層 ─► Softmax（3 類別）
```

- **音訊通道**：GRU → BiLSTM
- **文字通道**：BiLSTM
- **融合層**：拼接後送入全連接層與 Softmax 分類

---

## 🏋️ 模型訓練

- 損失函數：`CrossEntropyLoss`
- 優化器：`Adam`（lr=0.001）
- 訓練輪數：`10 epochs`
- 資料劃分：訓練集 80%，驗證集 20%
- 最佳模型儲存於：`model_weights.pth`

執行訓練：
```bash
python model_train.py
```

---

## 🤖 實作應用（語音互動版）

使用 OpenAI Whisper 進行語音辨識，結合模型進行情緒預測：

執行：
```bash
python chatbot.py
```

範例輸出：
```
🎤 請開始說話（5 秒）...
✅ 錄音完成！
📝 語音轉文字結果：我最近壓力有點大。
🔍 預測情緒為：負面
🤖 ChatBot 回應：你說的是「我最近壓力有點大」，我覺得你現在的情緒是「負面」。
```

---

## 🧪 CLI 單次推論版

提供手動輸入 `.wav` 與文字路徑的終端機互動：

```bash
python trained_bot.py
```

---

## 🧰 安裝與執行

### 1️⃣ 安裝環境
建議使用 Python 3.8+，並安裝依賴：
```bash
pip install -r requirements.txt
```

`requirements.txt` 範例：
```text
torch>=1.12.0
transformers>=4.30.0
librosa>=0.9.2
sounddevice>=0.4.4
whisper @ git+https://github.com/openai/whisper.git
```

---

## 📁 建議目錄結構

```
.
├── model_train.py         # 模型訓練程式
├── chatbot.py             # Whisper 語音互動推論
├── trained_bot.py         # CLI 推論版本
├── your_model.py          # 模型與特徵處理模組
├── model_weights.pth      # 已訓練模型權重
├── requirements.txt       # 相依套件
└── README.md              # 專案說明文件
```

---

## 🔄 模組說明

| 檔案              | 說明                          |
|-------------------|-------------------------------|
| `your_model.py`   | 模型結構與特徵萃取定義        |
| `model_train.py`  | 使用 EATD-Corpus 進行訓練     |
| `chatbot.py`      | Whisper 語音即時辨識與情緒預測 |
| `trained_bot.py`  | 手動輸入音訊與文字測試        |

---

## 🧩 延伸應用建議

- 📱 整合 LINE Bot 與 Web API（Flask / FastAPI）
- 📊 加入混淆矩陣與 F1 評估
- 🧬 擴充資料集、多語種支援

---

## 📌 結語

本專案展示了中文語音情緒辨識的完整流程，融合語音與文字模態，可應用於偏鄉心理照護、客服助理與行動裝置互動。未來可持續優化模型準確率並擴展應用場景。
