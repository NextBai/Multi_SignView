# Multi_SignViews 特徵與前處理系統技術文檔

## 📋 系統概述

Multi_SignViews 是一個基於**三模態融合**的手語辨識系統，支援30個常用手語詞彙的自動識別。系統整合視覺、音訊、文字三大模態，實現高精度的多模態手語理解。

### 🎯 核心技術特點
- **多模態融合**：視覺 + 音訊 + 文字三重特徵整合
- **高維特徵**：417維MediaPipe視覺特徵 + 89維光流 + 24維音訊
- **GPU加速**：支援Tesla P100、CUDA、MPS多種硬體加速
- **即時推論**：優化的特徵提取和融合管線
- **擴展性**：模組化設計，易於新增詞彙和模態

---

## 🏗️ 系統架構

### 資料夾結構
```
Multi_SignViews/
├── features/                          # 特徵資料總目錄
│   ├── mediapipe_features/            # MediaPipe視覺特徵
│   │   ├── again/                     # 詞彙別特徵資料夾
│   │   │   ├── again_001.npy          # 單一樣本特徵檔案
│   │   │   ├── again_002.npy
│   │   │   └── ... (~560個檔案)
│   │   ├── bird/
│   │   ├── book/
│   │   └── ... (30個詞彙)
│   ├── optical_flow_features/         # 光流運動特徵
│   ├── audio_features/                # TTS音訊特徵
│   ├── text_embeddings/               # 詞嵌入特徵
│   ├── semantic_features/             # 語意分析特徵
│   └── trimodal_mapping.json         # 三模態映射配置
├── 特徵及前處理/                        # 前處理程式碼
│   ├── mps_holistic_feature_extractor_optimized.py
│   ├── audio_feature_extractor.py
│   ├── multimodal_fusion.py
│   ├── features_integration_manager.py
│   └── ...
└── 其他專案檔案...
```

### 支援詞彙清單 (30個)
```
again, bird, book, computer, cousin, deaf, drink, eat, finish, fish,
friend, good, happy, learn, like, mother, need, nice, no, orange,
school, sister, student, table, teacher, tired, want, what, white, yes
```

---

## 📊 特徵資料規格

### 1. MediaPipe 視覺特徵
- **檔案格式**：`.npy` (NumPy陣列)
- **資料形狀**：`(100, 417)`
- **時間維度**：100個時間步 (約3.33秒@30fps)
- **特徵維度**：417維複合特徵向量
- **數值範圍**：[-4.40, 4.44] (標準化後)
- **資料類型**：`float32`

#### 特徵組成結構
| 模組 | 特徵點數量 | 維度 | 說明 |
|------|-----------|------|------|
| 手部特徵 | 21×2手 | 126維 | 雙手關節點3D座標 |
| 姿勢特徵 | 23個關鍵點 | 69維 | 身體姿態關鍵點 |
| 臉部特徵 | 74個表情點 | 222維 | 精細臉部表情特徵 |
| **總計** | **- ** | **417維** | **完整人體特徵** |

#### 重要特徵點定義
```python
# 姿勢關鍵點 (23個)
IMPORTANT_POSE_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,    # 頭部、軀幹
    11, 12, 13, 14, 15, 16,              # 肩膀、手肘、手腕
    17, 18, 19, 20, 21, 22               # 髖部、膝蓋、腳踝
]

# 臉部表情點 (74個)
IMPORTANT_FACE_INDICES = [
    # 眉毛 (10個): 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    # 眼睛 (16個): 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, ...
    # 鼻子 (23個): 1, 2, 5, 4, 6, 19, 94, 125, 141, 235, ...
    # 嘴巴 (22個): 61, 84, 17, 314, 405, 320, 307, 375, ...
    # 下巴 (3個): 172, 136, 150
]
```

### 2. 光流運動特徵
- **檔案格式**：`.npy`
- **資料形狀**：`(100, 89)`
- **特徵說明**：光流向量場，捕捉手勢運動軌跡
- **時間對齊**：與MediaPipe特徵同步

### 3. 音訊特徵
- **檔案格式**：`.npy`
- **資料形狀**：`(24,)`
- **特徵組成**：MFCC(13) + Spectral(7) + Temporal(4) = 24維
- **融合變體**：
  - `*_max_fusion_24d.npy` - 最大值融合
  - `*_mean_fusion_24d.npy` - 平均值融合
  - `*_normalized_24d.npy` - 標準化融合
  - `*_weighted_fusion_24d.npy` - 權重融合

### 4. 文字嵌入特徵
- **模型類型**：Word2Vec、FastText、BERT、統一嵌入
- **檔案格式**：`.npy`
- **用途**：語意理解和跨模態對齊

### 5. 三模態映射配置
**檔案**：`trimodal_mapping.json`
```json
{
  "again": {
    "word": "again",
    "modalities": {
      "visual": {
        "mediapipe_dir": "features/mediapipe_features/again",
        "optical_flow_dir": "features/optical_flow_features/again",
        "available": true
      },
      "audio": {
        "features_json": "features/audio_features/again_audio_features.json",
        "fusion_variants": [...],
        "available": true
      },
      "text": {
        "embeddings_dir": "features/text_embeddings",
        "semantic_file": "features/semantic_features/semantic_analysis_fixed.json",
        "available": true
      }
    },
    "cross_modal_links": {
      "visual_audio_alignment": "video_again ↔ audio_again",
      "visual_text_mapping": "video_again ↔ text_again",
      "audio_text_correspondence": "audio_again ↔ text_again",
      "trimodal_fusion": "video_again + audio_again + text_again"
    }
  },
  // ... 其他29個詞彙
}
```

---

## 🔧 核心處理模組

### 1. MediaPipe特徵提取器
**檔案**：`mps_holistic_feature_extractor_optimized.py`

#### 主要類別：`OptimizedMpsHolisticExtractor`
```python
class OptimizedMpsHolisticExtractor:
    def __init__(self,
                 target_fps: int = 30,           # 目標影格率
                 target_frames: int = 100,       # 標準化幀數
                 confidence_threshold: float = 0.3,  # 置信度閾值
                 use_gpu: bool = True,           # GPU加速
                 normalize_method: str = "standard",  # 正規化方法
                 enable_tracking: bool = True,   # 追蹤優化
                 batch_processing: bool = True): # 批次處理
```

#### 技術特點
- **GPU加速**：自動偵測Tesla P100、CUDA、MPS硬體
- **追蹤優化**：2025年版本，1.5x速度提升
- **記憶體管理**：16GB GPU記憶體優化
- **多進程並行**：批次處理大量視頻檔案
- **錯誤恢復**：模型自動下載和驗證機制

#### 核心方法
```python
def extract_features_from_video(self, video_path: str) -> np.ndarray:
    """從視頻提取標準化特徵 (100, 417)"""

def _extract_landmarks_from_frame(self, frame: np.ndarray, timestamp_ms: int):
    """單幀特徵提取"""

def _normalize_features(self, features: np.ndarray) -> np.ndarray:
    """特徵標準化處理"""
```

### 2. 音訊特徵提取器
**檔案**：`audio_feature_extractor.py`

#### 主要功能
- **TTS語音生成**：自動產生30個英文詞彙的標準發音
- **多樣化語音**：不同口音、語速、性別的語音變體
- **特徵提取**：MFCC、頻譜、時域特徵計算
- **品質評估**：語音清晰度和一致性檢查
- **融合策略**：四種不同的特徵融合方法

#### 特徵計算流程
```python
# 1. 語音預處理
audio_data = librosa.load(audio_file, sr=22050)
audio_data = self._preprocess_audio(audio_data)

# 2. MFCC特徵 (13維)
mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)

# 3. 頻譜特徵 (7維)
spectral_features = self._extract_spectral_features(audio_data)

# 4. 時域特徵 (4維)
temporal_features = self._extract_temporal_features(audio_data)

# 5. 特徵融合 (24維)
fused_features = np.concatenate([mfcc_stats, spectral_features, temporal_features])
```

### 3. 多模態融合器
**檔案**：`multimodal_fusion.py`

#### 融合架構
```python
class MultiModalFusionNetwork(nn.Module):
    def __init__(self):
        # 視覺編碼器: 417 → 256
        self.visual_encoder = nn.Sequential(...)

        # 音訊編碼器: 24 → 64
        self.audio_encoder = nn.Sequential(...)

        # 文字編碼器: 300 → 128
        self.text_encoder = nn.Sequential(...)

        # 跨模態注意力機制
        self.cross_modal_attention = MultiHeadCrossAttention(...)

        # 最終融合層: 448 → 256
        self.fusion_layer = nn.Sequential(...)
```

#### 融合策略
1. **早期融合**：特徵層面直接拼接
2. **中期融合**：注意力權重調節
3. **晚期融合**：決策層面投票機制
4. **自適應融合**：動態模態權重學習

### 4. 特徵整合管理器
**檔案**：`features_integration_manager.py`

#### 主要功能
- **完整性檢查**：驗證所有模態特徵檔案存在性
- **格式驗證**：確保特徵維度和資料類型正確
- **統計分析**：特徵分佈和品質評估
- **批次載入**：高效的特徵資料載入接口

---

## ⚡ 性能優化

### GPU加速配置
```python
# Tesla P100 專用優化
if "P100" in device_name:
    print("⚡ 啟用 Tesla P100 專用優化")
    delegate = BaseOptions.Delegate.GPU
    # 16GB 記憶體優化設定

# Apple Silicon MPS
elif torch.backends.mps.is_available():
    print("🚀 檢測到Apple Silicon MPS")
    delegate = BaseOptions.Delegate.GPU
```

### 記憶體管理
- **批次大小自動調節**：根據可用GPU記憶體動態調整
- **特徵快取機制**：避免重複計算相同檔案
- **進程池管理**：多核心並行處理大量檔案

### 處理速度
- **單個視頻**：~2-3秒 (Tesla P100)
- **批次處理**：~1.5x 速度提升 (啟用追蹤優化)
- **總數據量**：30詞 × 560樣本 = 16,800個多模態樣本

---

## 📈 資料統計

### 樣本分佈
- **每個詞彙**：平均560個樣本檔案
- **總樣本數**：~16,800個多模態樣本
- **檔案大小**：
  - MediaPipe特徵：~167KB/檔案 (100×417×4字節)
  - 光流特徵：~36KB/檔案 (100×89×4字節)
  - 音訊特徵：~96字節/檔案 (24×4字節)

### 儲存需求
- **MediaPipe特徵**：~2.8GB (16,800 × 167KB)
- **光流特徵**：~600MB (16,800 × 36KB)
- **音訊特徵**：~1.6MB (16,800 × 96B)
- **總計**：~3.4GB

---

## 🔬 技術優勢

### 1. 高精度特徵表示
- **417維MediaPipe特徵**：涵蓋手部、姿勢、臉部完整資訊
- **時序建模**：100幀標準化時間序列
- **多模態互補**：視覺、音訊、文字三重驗證

### 2. 系統魯棒性
- **缺失模態補償**：單一模態失效仍可運作
- **品質控制**：置信度閾值和異常值檢測
- **標準化處理**：跨樣本特徵一致性保證

### 3. 擴展性設計
- **模組化架構**：獨立的特徵提取和融合模組
- **配置驅動**：JSON配置檔案管理詞彙和路徑
- **介面統一**：標準化的資料載入和處理接口

### 4. 即時性能
- **GPU加速**：支援多種硬體平台
- **並行處理**：多進程特徵提取
- **快取機制**：避免重複計算

---

## 🚀 應用場景

### 1. 手語翻譯系統
- **即時手語辨識**：攝影機輸入 → 多模態特徵 → 詞彙識別
- **批次影片處理**：手語影片資料庫分析
- **教學輔助**：手語學習和評估系統

### 2. 研究開發
- **特徵分析**：手語動作的量化研究
- **模型訓練**：深度學習模型的特徵輸入
- **跨模態研究**：多模態融合演算法開發

### 3. 擴展應用
- **新詞彙新增**：遵循現有架構新增手語詞彙
- **多語言支援**：擴展至其他手語系統
- **客製化需求**：針對特定領域的手語識別

---

## 📚 使用指南

### 環境需求
```bash
# Python 3.8+
pip install mediapipe opencv-python numpy scipy
pip install librosa soundfile gtts pydub
pip install torch torchvision  # GPU版本
pip install tqdm pandas scikit-learn
```

### 基本使用
```python
from 特徵及前處理.features_integration_manager import FeaturesIntegrationManager

# 初始化管理器
manager = FeaturesIntegrationManager("features/")

# 檢查資料完整性
status = manager.check_directory_structure()

# 載入特定詞彙的特徵
features = manager.load_word_features("again", modalities=["visual", "audio"])
```

### 特徵提取
```python
from 特徵及前處理.mps_holistic_feature_extractor_optimized import OptimizedMpsHolisticExtractor

# 初始化提取器
extractor = OptimizedMpsHolisticExtractor(use_gpu=True)

# 提取視頻特徵
features = extractor.extract_features_from_video("path/to/video.mp4")
print(f"特徵形狀: {features.shape}")  # (100, 417)
```

---

## 🔧 故障排除

### 常見問題
1. **GPU記憶體不足**：降低批次大小或使用CPU模式
2. **模型檔案缺失**：系統會自動下載MediaPipe模型
3. **特徵維度不符**：檢查視頻品質和MediaPipe偵測結果
4. **音訊TTS失敗**：確保網路連接和TTS服務可用

### 效能調優
```python
# 記憶體優化
extractor = OptimizedMpsHolisticExtractor(
    batch_processing=True,      # 啟用批次處理
    enable_tracking=True,       # 啟用追蹤優化
    confidence_threshold=0.5    # 提高置信度閾值
)
```

---

## 📄 授權與貢獻

**開發團隊**：Claude Code + Multi_SignView Team
**開發時間**：2024-2025
**技術支援**：MediaPipe、PyTorch、LibROSA

這套系統為手語辨識研究提供了完整的多模態特徵提取和融合解決方案，具備高精度、高效能、易擴展的特點，適用於學術研究和實際應用開發。

---

*最後更新：2025年9月*