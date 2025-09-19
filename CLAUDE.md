# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

Multi_SignView 是一個**真正的多模態手語辨識系統**，整合視覺、音訊、文字三大模態特徵，實現端到端的跨模態手語理解與生成。

**核心特色**
- **三模態完整覆蓋**：視覺手語動作 + 音訊發音 + 文字語義，30個詞彙完整特徵庫
- **跨模態對應關係**：每個詞彙建立視覺↔音訊、視覺↔文字、音訊↔文字的完整映射
- **多重應用場景**：手語學習、跨模態檢索、多模態識別、教學輔助等6大用途

**重要：本專案嚴格遵守 TDD（測試驅動開發）原則**
- 所有開發必須先撰寫文檔，再進行代碼實作
- 遵循「紅燈 → 綠燈 → 重構」循環：先寫測試、再寫實作、最後優化
- 任何新功能都必須有對應的單元測試

## 專案架構

### 核心模組結構
- `特徵及前處理/` - 多模態特徵提取和前處理模組
  - **視覺特徵模組**
    - `mps_holistic_feature_extractor_optimized.py` - MediaPipe 整體特徵提取器（手部、姿態、臉部）
    - `mps_optical_flow_extractor.py` - 光流運動特徵提取器
    - `selfie_segmentation.py` - 人像分割功能
  - **音訊特徵模組**
    - `audio_feature_extractor.py` - 音訊多維特徵提取器（MFCC、Chroma、Spectral）
    - `audio_preprocessing.py` - 音訊預處理和時序對齊
  - **文字特徵模組**
    - `text_feature_extractor.py` - 文字語義特徵提取器（詞嵌入、語義編碼）
    - `vocabulary_manager.py` - 詞彙表管理和語義分析
  - **多模態融合模組**
    - `multimodal_fusion.py` - 多模態特徵融合和注意力機制
    - `cross_modal_attention.py` - 跨模態注意力計算
  - `model/` - 預訓練模型檔案存放目錄

- `bai_dataset/` - 原始手語資料集，包含30個手語詞彙類別
  - 每個詞彙目錄包含多個影片檔案，**資料數量不均衡**

- `bai_datasets/` - 經過數據平衡和去背處理後的影片資料夾
  - 由 `bai_dataset` 經過 `selfie_segmentation.py` 去背處理後產生
  - 已進行類別平衡處理，可直接用於特徵提取

- `features/` - **多模態特徵統一管理目錄**（✅ 完整覆蓋30個詞彙）
  - **視覺特徵**
    - `mediapipe_features/` - MediaPipe 整體特徵提取結果（手部+姿態+臉部）
    - `optical_flow_features/` - 光流運動特徵提取結果（3種演算法）
  - **音訊特徵**
    - `audio_features/` - TTS詞彙音訊特徵（24維MFCC+Spectral+Temporal）
    - 每詞彙包含：4種口音變體 × 5種融合方式 = 20個音訊檔案
  - **文字特徵**
    - `text_embeddings/` - 4種詞嵌入（Word2Vec+FastText+BERT+Unified）
    - `semantic_features/` - 語義分析和相似度矩陣
  - **整合管理檔案**
    - `integration_report.json` - 特徵完整性檢查報告
    - `trimodal_mapping.json` - 三模態檔案路徑對應表（重要！）

## 開發環境設定

### Python 環境
- 使用 Python 3.x
- **視覺處理依賴**：
  - `mediapipe` - Google MediaPipe 框架
  - `opencv-python` (cv2) - 影像處理
- **音訊處理依賴**：
  - `librosa` - 音訊分析和特徵提取
  - `soundfile` - 音訊檔案讀寫
  - `scipy` - 科學計算和信號處理
- **文字處理依賴**：
  - `gensim` - Word2Vec 和 FastText 詞嵌入
  - `transformers` - BERT/RoBERTa 預訓練模型
  - `sentence-transformers` - 句子級語義編碼
  - `nltk` - 自然語言處理工具包
- **多模態融合依賴**：
  - `torch` - PyTorch 深度學習框架
  - `sklearn` - 機器學習工具包
- **通用依賴**：
  - `numpy` - 數值計算
  - `pandas` - 數據處理
  - `tqdm` - 進度條顯示
  - `matplotlib` - 數據視覺化

### 執行特徵提取

**重要：多模態特徵提取流程（✅ 已完成）**
1. **資料前處理** - `bai_dataset/` → `bai_datasets/`（去背+平衡）
2. **視覺特徵提取** - `bai_datasets/` → `features/mediapipe_features/ + optical_flow_features/`
3. **音訊特徵提取** - TTS下載 → `features/audio_features/`（30詞彙×20檔案）
4. **文字特徵提取** - 詞彙語義 → `features/text_embeddings/ + semantic_features/`
5. **特徵整合管理** - 生成三模態對應關係 → `features/trimodal_mapping.json`

```bash
# 視覺特徵提取（已完成）
python3 特徵及前處理/mps_holistic_feature_extractor_optimized.py
python3 特徵及前處理/mps_optical_flow_extractor.py

# 音訊特徵提取（已完成）
python3 audio_feature_extractor.py

# 文字特徵提取（已完成）
python3 text_feature_extractor.py

# 語義特徵修復（已完成）
python3 fix_semantic_features.py

# 特徵完整性檢查
python3 features_integration_manager.py
```

**多模態資料載入範例**
```python
import json
import numpy as np

# 載入三模態對應關係
with open('features/trimodal_mapping.json', 'r') as f:
    mapping = json.load(f)

# 載入特定詞彙的三模態特徵
word = "again"
word_data = mapping[word]

# 視覺特徵
visual_files = f"features/mediapipe_features/{word}/*.npy"
# 音訊特徵
audio_feat = np.load(f"features/audio_features/{word}_normalized_24d.npy")
# 文字特徵
text_feat = np.load("features/text_embeddings/unified_embeddings.npy")[0]
```

## 技術特點

### MediaPipe 特徵提取器技術細節

**核心架構 (`mps_holistic_feature_extractor_optimized.py`)**
- **OptimizedMpsHolisticExtractor 類別**：整合手部、姿態、臉部三大模態特徵提取
- **自動模型管理**：支援 MediaPipe 官方模型自動下載與 Hash 驗證
- **相容性處理**：支援不同版本 MediaPipe API 的動態載入機制
- **移除 Stub 模式**：避免不穩定的實驗性功能，提升穩定性

**多模態特徵提取能力**
- **手部特徵 (HandLandmarker)**：
  - 21個關鍵點 x 2隻手 = 42個 3D 座標點
  - 支援手部姿態分類和手勢識別
  - 包含手掌朝向、關節角度等幾何特徵
- **姿態特徵 (PoseLandmarker)**：
  - 33個身體關鍵點的 3D 座標
  - 涵蓋頭部、軀幹、四肢完整骨架結構
  - 提供身體姿態角度和重心資訊
- **臉部特徵 (FaceLandmarker)**：
  - 468個臉部關鍵點的精密定位
  - 包含眼部、鼻部、嘴部、輪廓等細節
  - 支援表情分析和口型識別

**技術優化特色**
- **Tesla P100 GPU 特化**：針對 Kaggle 環境進行多執行緒調整
- **特徵正規化**：自動進行座標標準化和異常值處理
- **錯誤恢復機制**：robustness 處理影片讀取失敗和特徵提取異常
- **批次處理支援**：使用 ProcessPoolExecutor 進行多進程並行處理
- **記憶體優化**：控制 OpenMP/OpenBLAS 執行緒數避免資源競爭

### 光流特徵提取器技術細節

**核心架構 (`mps_optical_flow_extractor.py`)**
- **OpticalFlowExtractor 類別**：多演算法光流運動特徵提取器
- **KaggleOpticalFlowExtractor 類別**：Kaggle 環境最佳化版本
- **MpsOpticalFlowExtractor 類別**：與 MediaPipe 整合的包裝器

**三大光流演算法支援**
- **Farneback 光流 (`_extract_farneback_features`)**：
  - 基於多項式展開的稠密光流計算
  - 適合全域運動分析和背景運動檢測
  - 提供像素級運動向量場和運動強度分布
- **Lucas-Kanade 光流 (`_extract_lucas_kanade_features`)**：
  - 基於特徵點追蹤的稀疏光流
  - 專注於關鍵運動點的高精度追蹤
  - 支援 Shi-Tomasi 角點檢測和運動軌跡分析
- **TVL1 光流 (`_extract_tvl1_features`)**：
  - 基於變分法的精確光流估計
  - 提供亞像素級運動精度
  - 適合複雜運動模式和遮擋處理

**動態 ROI 檢測技術**
- **自適應 ROI 偵測**：根據運動強度動態調整感興趣區域
- **運動熱點分析**：識別影片中運動最活躍的區域
- **多尺度運動分析**：支援不同空間尺度的運動特徵提取
- **ROI 特徵聚合**：將區域內運動特徵進行統計聚合

**特徵工程與正規化**
- **時序正規化**：將不同長度影片統一到100幀標準長度
- **運動向量分解**：提取運動幅度、方向、加速度等多維特徵
- **統計特徵計算**：包含均值、標準差、最大值、能量等統計量
- **特徵序列建構**：生成時間相關的運動特徵序列

**性能最佳化設計**
- **多進程架構**：預設2個 worker 進程避免資源競爭
- **記憶體管理**：控制幀數據暫存和批次處理大小
- **執行緒控制**：限制 OpenCV 和數值庫執行緒數
- **進度追蹤**：使用 tqdm 提供詳細處理進度資訊

### 音訊特徵提取器技術細節

**核心架構 (`audio_feature_extractor.py`)**
- **VocabularyAudioDownloader 類別**：TTS音訊自動下載器
- **AudioFeatureExtractor 類別**：多維音訊特徵統一提取器
- **多口音支援**：美式、英式、澳式、慢速4種變體下載

**TTS音訊庫構建**
- **30個詞彙 × 4種口音變體 = 120個音訊檔案**
- **音訊來源**：Google TTS (gTTS) 高品質語音合成
- **採樣率標準化**：統一16kHz單聲道格式
- **存放位置**：`特徵及前處理/vocabulary_audio/詞彙/`

**24維音訊特徵提取**
- **MFCC 特徵 (13維度)**：
  - Mel-frequency cepstral coefficients 倒頻譜係數
  - 捕捉語音頻譜包絡，表示發音特徵
  - 對不同口音具有一定魯棒性
- **Spectral 特徵 (7維度)**：
  - Spectral Centroid：頻譜重心，反映音色亮度
  - Spectral Bandwidth：頻譜帶寬，描述頻率分布
  - Spectral Rolloff：頻譜滾降點，能量集中度
  - Zero Crossing Rate：過零率，音頻週期性
  - RMS Energy：均方根能量，音量強度
  - Spectral Contrast：頻譜對比度，諧波特徵
  - Chroma：色度特徵，音高資訊
- **Temporal 特徵 (4維度)**：
  - Tempo：節拍速度檢測
  - Onset Rate：起始點密度
  - Duration：音訊總時長
  - Silence Ratio：靜音比例

**5種特徵融合策略**
- **mean_fusion**：4種口音特徵的平均值融合
- **max_fusion**：取各維度的最大值
- **weighted_fusion**：加權平均（美式0.4，英式0.3，澳式0.2，慢速0.1）
- **normalized**：L2正規化後的標準特徵
- **std_across_versions**：跨口音變體的標準差特徵

**實際應用範例**
```python
# 載入特定詞彙的音訊特徵
word = "again"
audio_features = {
    'mean': np.load(f"features/audio_features/{word}_mean_fusion_24d.npy"),
    'weighted': np.load(f"features/audio_features/{word}_weighted_fusion_24d.npy"),
    'normalized': np.load(f"features/audio_features/{word}_normalized_24d.npy")
}
```

### 文字特徵提取器技術細節

**核心架構 (`text_feature_extractor.py`)**
- **TextFeatureExtractor 類別**：多層次文字語義特徵提取器
- **VocabularyManager 類別**：30個手語詞彙的語義管理系統
- **多語言支援**：中英文詞彙混合處理能力

**多層次詞嵌入技術**
- **Word2Vec 嵌入 (300維度)**：
  - Skip-gram 模型訓練的詞向量
  - 捕捉詞彙語義相似性和語言學關係
  - 支援詞彙語義距離計算和相似詞查找
- **FastText 嵌入 (300維度)**：
  - 子詞資訊增強的詞向量表示
  - 處理未登錄詞 (OOV) 和詞形變化
  - 適合處理手語詞彙的多樣性表達
- **BERT 語義編碼 (768維度)**：
  - 上下文感知的預訓練語言模型
  - Transformer 架構的深層語義理解
  - 支援句子級和詞彙級的語義表示

**語義增強與分析 (`vocabulary_manager.py`)**
- **詞彙語義聚類**：
  - 30個手語詞彙的語義空間分布分析
  - K-means 聚類識別語義相近的詞彙群組
  - 動作類 (again, finish)、物品類 (book, table)、人物類 (mother, teacher) 等語義分類
- **語義相似度矩陣**：
  - 30×30 詞彙相似度關係矩陣計算
  - 餘弦相似度和歐氏距離雙重測量
  - 支援語義導向的分類性能優化
- **同義詞擴展**：
  - WordNet 和自定義詞典的語義擴展
  - 語義網絡構建和關聯詞彙發現
  - 增強小樣本學習的語義泛化能力

### 多模態融合技術細節

**核心架構 (`multimodal_fusion.py`)**
- **MultiModalFusion 類別**：三模態統一融合框架
- **AdaptiveWeighting 機制**：動態模態權重自適應調整
- **ModalityAlignment 模組**：跨模態特徵對齊和標準化

**多層次融合策略**
- **特徵層融合 (Early Fusion)**：
  - 視覺特徵 (512維) + 音訊特徵 (36維) + 文字特徵 (300維) = 848維統一特徵
  - 加權拼接和維度壓縮，PCA 降維到256維核心表示
  - 模態間特徵正規化，消除量綱差異影響
- **表示層融合 (Mid-level Fusion)**：
  - 三模態獨立編碼後的中層表示融合
  - 跨模態特徵交互學習，捕捉模態間互補資訊
  - 多頭注意力機制分配模態重要性權重
- **決策層融合 (Late Fusion)**：
  - 三個模態分別訓練獨立分類器
  - 集成學習方法 (Voting, Stacking) 綜合決策
  - 模態置信度加權投票機制

**跨模態注意力機制 (`cross_modal_attention.py`)**
- **Multi-Head Cross-Modal Attention**：
  - 視覺→音訊、視覺→文字、音訊→文字三對交互注意力
  - Query-Key-Value 架構計算跨模態依賴關係
  - 8個注意力頭並行處理，捕捉多層次交互模式
- **時序注意力對齊**：
  - 影像100幀與音訊時間片段的動態對齊
  - 文字語義與影像動作序列的語義對應
  - Soft Attention 機制處理時序不一致問題
- **自適應模態權重**：
  - 不同手語詞彙的模態重要性自動學習
  - 動作類詞彙偏重視覺，音調類偏重音訊
  - 語義複雜詞彙增強文字模態權重

**魯棒性設計**
- **缺失模態處理**：
  - 模態缺失檢測和補償機制
  - 單模態、雙模態、三模態靈活推理
  - 模態缺失時的性能優雅降級
- **噪音容忍性**：
  - 音訊降噪和異常值檢測
  - 視覺特徵魯棒性增強
  - 多模態一致性約束正規化訓練

### 資料集結構

**原始資料集 (`bai_dataset/`)**
- 30個手語詞彙類別
- **資料數量不均衡** - 各類別影片數量差異較大，從30-100個樣本不等
- 詞彙包括：again, bird, book, computer, cousin, deaf, drink, eat, finish, fish, friend, good, happy, learn, like, mother, need, nice, no, orange, school, sister, student, table, teacher, tired, want, what, white, yes

**處理後資料集 (`bai_datasets/`)**
- 經過去背和數據平衡處理
- 可直接用於雙流特徵提取
- 已解決類別不均衡問題

**多模態特徵資料 (`features/`) - ✅ 已完成提取**
- **視覺特徵**
  - `mediapipe_features/` - 手部(21×2)、姿態(33)、臉部(468) 3D座標（30詞彙×560檔案）
  - `optical_flow_features/` - Farneback/Lucas-Kanade/TVL1 光流運動特徵（30詞彙）
- **音訊特徵**
  - `audio_features/` - MFCC(13) + Spectral(7) + Temporal(4) = 24維特徵
  - 每詞彙6檔案：JSON詳情 + 5種融合.npy，共180個音訊特徵檔案
- **文字特徵**
  - `text_embeddings/` - Word2Vec(300) + FastText(300) + BERT(768) + Unified(300)
  - `semantic_features/semantic_analysis_fixed.json` - 語義分類和相似度矩陣
- **管理檔案**
  - `integration_report.json` - 完整性檢查報告
  - `trimodal_mapping.json` - 三模態檔案路徑索引（載入必備！）

**資料統計**
- **總詞彙**: 30個手語詞彙
- **視覺檔案**: ~16,800個MediaPipe特徵檔案 + 30個光流目錄
- **音訊檔案**: 180個特徵檔案（30詞彙×6檔案）
- **文字檔案**: 4個嵌入矩陣 + 1個語義分析
- **特徵完整度**: 100%覆蓋率

## 性能優化設定

### 多進程配置
- 預設使用2個 worker 進程避免資源競爭
- 限制 OpenMP 和 OpenBLAS 執行緒數為1
- CV2 執行緒數設為0

### 記憶體管理
- 使用 tqdm 進度條追蹤處理狀態
- 實作錯誤處理機制避免進程崩潰
- 支援大批次資料處理

## 模型管理
- 模型檔案自動下載至 `特徵及前處理/model/` 目錄
- 支援 MediaPipe 官方模型文件的 Hash 驗證
- 相容性處理支援不同版本的 MediaPipe

## 使用說明

### 多模態模型訓練
```python
import json
import numpy as np
from pathlib import Path

# 1. 載入三模態映射關係
with open('features/trimodal_mapping.json', 'r') as f:
    trimodal_data = json.load(f)

# 2. 批次載入所有詞彙的特徵
def load_multimodal_features():
    features = {'visual': [], 'audio': [], 'text': []}
    labels = []

    for i, (word, data) in enumerate(trimodal_data.items()):
        # 視覺特徵 (選擇一個檔案作為代表)
        visual_dir = Path(data['modalities']['visual']['mediapipe_dir'])
        visual_file = list(visual_dir.glob('*.npy'))[0]
        visual_feat = np.load(visual_file)

        # 音訊特徵 (使用normalized版本)
        audio_feat = np.load(f"features/audio_features/{word}_normalized_24d.npy")

        # 文字特徵 (使用unified嵌入)
        text_feat = np.load("features/text_embeddings/unified_embeddings.npy")[i]

        features['visual'].append(visual_feat)
        features['audio'].append(audio_feat)
        features['text'].append(text_feat)
        labels.append(i)  # 詞彙索引作為標籤

    return features, labels
```

### 跨模態檢索範例
```python
# 根據音訊特徵檢索對應的手語動作
def audio_to_sign_retrieval(query_audio_features):
    similarities = []
    for word in trimodal_data.keys():
        stored_audio = np.load(f"features/audio_features/{word}_normalized_24d.npy")
        similarity = np.cosine_similarity([query_audio_features], [stored_audio])[0][0]
        similarities.append((word, similarity))

    # 返回最相似的手語詞彙
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
```

## vocabulary_audio 的六大用途

1. **語音參考資源** - 手語學習者的標準發音參考
2. **跨模態學習** - 視覺-音訊模態對比學習訓練
3. **多模態檢索** - 語音→手語、手語→語音的雙向檢索
4. **教學與評估** - 手語教學中的語音提示和同步性評估
5. **系統擴展** - 整合語音識別和語音合成系統
6. **研究應用** - 跨模態注意力機制和多模態融合算法研究

## 注意事項
- 專案針對 Kaggle Tesla P100 GPU 環境優化
- 建議在有 GPU 支援的環境下執行以提升處理速度
- 多模態資料總大小約2-3GB，需預留足夠磁碟空間
- 使用 `trimodal_mapping.json` 作為資料載入的統一索引檔案