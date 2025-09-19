# 多模態融合LLM訓練指南

## 概述

本指南基於Multi_SignViews專案的features資料夾，提供完整的**三模態手語識別LLM**訓練方案。整合視覺手語動作、音訊發音、文字語義三大模態，實現端到端的跨模態手語理解與生成。

## 資料特徵分析

### 特徵維度總覽

| 模態 | 特徵類型 | 維度 | 檔案格式 | 特點 |
|------|----------|------|----------|------|
| **視覺** | MediaPipe骨架 | (100, 417) | .npy | 時序特徵，100幀×417維全身骨架 |
| **音訊** | 多維音訊特徵 | (24,) | .npy | MFCC+Spectral+Temporal，4種融合方式 |
| **文字** | 多種詞嵌入 | (300/768,) | .npy | Word2Vec/FastText/BERT/Unified |

### 資料規模
- **30個手語詞彙**：完整詞彙表覆蓋
- **視覺檔案**：~16,800個MediaPipe特徵檔案（平均每詞彙560個）
- **音訊檔案**：180個特徵檔案（30詞彙×6變體）
- **文字檔案**：4個嵌入矩陣 + 語義分析
- **管理檔案**：`trimodal_mapping.json` 提供完整對應關係

## 多模態LLM架構設計

### 1. 整體架構 (Trimodal Transformer)

```
Input Layer:
┌─────────────┬─────────────┬─────────────┐
│ 視覺序列     │ 音訊向量     │ 文字嵌入     │
│ (100, 417)  │ (24,)       │ (300/768,)  │
└─────────────┴─────────────┴─────────────┘
           │         │         │
┌─────────────┬─────────────┬─────────────┐
│ Visual      │ Audio       │ Text        │
│ Encoder     │ Encoder     │ Encoder     │
│ (Temporal)  │ (Dense)     │ (Semantic)  │
└─────────────┴─────────────┴─────────────┘
           │         │         │
┌─────────────────────────────────────────┐
│     Cross-Modal Attention Layer        │
│   (視覺↔音訊, 視覺↔文字, 音訊↔文字)     │
└─────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────┐
│        Multimodal Fusion Layer         │
│     (Adaptive Weighted Combination)     │
└─────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────┐
│         Transformer Decoder            │
│      (GPT-style Language Model)        │
└─────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────┐
│            Output Layer               │
│  分類: 30類手語 | 生成: 文字描述        │
└─────────────────────────────────────────┘
```

### 2. 模態編碼器設計

#### 視覺編碼器 (Temporal Visual Encoder)
```python
class VisualEncoder(nn.Module):
    def __init__(self, input_dim=417, hidden_dim=512, num_frames=100):
        super().__init__()
        self.frame_embedding = nn.Linear(input_dim, hidden_dim)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=num_frames)

    def forward(self, x):  # x: (batch, 100, 417)
        # 時序特徵嵌入
        x = self.frame_embedding(x)  # (batch, 100, 512)
        x = self.position_encoding(x)
        # Transformer時序編碼
        x = self.temporal_encoder(x)  # (batch, 100, 512)
        return x
```

#### 音訊編碼器 (Audio Feature Encoder)
```python
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=512):
        super().__init__()
        self.audio_projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, x):  # x: (batch, 24)
        return self.audio_projector(x)  # (batch, 512)
```

#### 文字編碼器 (Semantic Text Encoder)
```python
class TextEncoder(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512):
        super().__init__()
        self.text_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):  # x: (batch, 300)
        return self.text_projector(x)  # (batch, 512)
```

### 3. 跨模態注意力機制

```python
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, query, key, value):
        # 跨模態注意力計算
        attn_output, attn_weights = self.multihead_attn(
            query, key, value
        )
        return attn_output, attn_weights

class TrimodalFusion(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        # 三對跨模態注意力
        self.visual_audio_attn = CrossModalAttention(hidden_dim)
        self.visual_text_attn = CrossModalAttention(hidden_dim)
        self.audio_text_attn = CrossModalAttention(hidden_dim)

        # 自適應權重學習
        self.modal_weights = nn.Parameter(torch.ones(3))
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, visual_feat, audio_feat, text_feat):
        # visual_feat: (batch, 100, 512)
        # audio_feat: (batch, 512) -> (batch, 1, 512)
        # text_feat: (batch, 512) -> (batch, 1, 512)

        audio_feat = audio_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(1)

        # 跨模態交互
        va_fusion, _ = self.visual_audio_attn(visual_feat, audio_feat, audio_feat)
        vt_fusion, _ = self.visual_text_attn(visual_feat, text_feat, text_feat)
        at_fusion, _ = self.audio_text_attn(audio_feat, text_feat, text_feat)

        # 時序池化
        va_pooled = va_fusion.mean(dim=1)  # (batch, 512)
        vt_pooled = vt_fusion.mean(dim=1)  # (batch, 512)
        at_pooled = at_fusion.mean(dim=1)  # (batch, 512)

        # 自適應權重融合
        weights = F.softmax(self.modal_weights, dim=0)
        fused = torch.cat([
            weights[0] * va_pooled,
            weights[1] * vt_pooled,
            weights[2] * at_pooled
        ], dim=-1)

        return self.fusion_layer(fused)  # (batch, 512)
```

### 4. 完整多模態LLM

```python
class TrimodalSignLanguageLLM(nn.Module):
    def __init__(self, vocab_size=30, hidden_dim=512):
        super().__init__()

        # 模態編碼器
        self.visual_encoder = VisualEncoder()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()

        # 跨模態融合
        self.trimodal_fusion = TrimodalFusion(hidden_dim)

        # 語言模型頭
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, vocab_size)
        )

        # 文字生成頭 (可選)
        self.text_generation_head = nn.Linear(hidden_dim, 50257)  # GPT vocabulary

    def forward(self, visual_seq, audio_feat, text_embed, task='classification'):
        # 模態編碼
        visual_encoded = self.visual_encoder(visual_seq)
        audio_encoded = self.audio_encoder(audio_feat)
        text_encoded = self.text_encoder(text_embed)

        # 跨模態融合
        fused_repr = self.trimodal_fusion(
            visual_encoded, audio_encoded, text_encoded
        )

        # 任務特定輸出
        if task == 'classification':
            return self.lm_head(fused_repr)
        elif task == 'generation':
            return self.text_generation_head(fused_repr)
        else:
            return fused_repr
```

## 訓練策略

### 1. 多階段訓練策略

#### 階段一：單模態預訓練 (Modality-Specific Pre-training)
```python
# 目標：每個模態獨立學習詞彙特徵表示
# 時程：10-15 epochs
# 學習率：1e-4

# 視覺模態：時序動作識別
visual_loss = CrossEntropyLoss(visual_logits, labels)

# 音訊模態：音訊特徵分類
audio_loss = CrossEntropyLoss(audio_logits, labels)

# 文字模態：語義嵌入分類
text_loss = CrossEntropyLoss(text_logits, labels)

total_loss = visual_loss + audio_loss + text_loss
```

#### 階段二：跨模態對齊訓練 (Cross-Modal Alignment)
```python
# 目標：學習模態間的對應關係
# 時程：15-20 epochs
# 學習率：5e-5

# 對比學習：正樣本(同詞彙不同模態)，負樣本(不同詞彙)
def contrastive_loss(visual_feat, audio_feat, text_feat, labels):
    # 計算模態間相似度
    va_sim = cosine_similarity(visual_feat, audio_feat)
    vt_sim = cosine_similarity(visual_feat, text_feat)
    at_sim = cosine_similarity(audio_feat, text_feat)

    # 對比損失
    loss = contrastive_criterion(va_sim, labels) + \
           contrastive_criterion(vt_sim, labels) + \
           contrastive_criterion(at_sim, labels)
    return loss
```

#### 階段三：端到端聯合訓練 (End-to-End Joint Training)
```python
# 目標：整體性能優化
# 時程：20-30 epochs
# 學習率：1e-5

# 多任務聯合損失
classification_loss = CrossEntropyLoss(multimodal_logits, labels)
alignment_loss = contrastive_loss(visual_feat, audio_feat, text_feat, labels)
regularization_loss = modal_weights_regularization()

total_loss = classification_loss + 0.1 * alignment_loss + 0.01 * regularization_loss
```

### 2. 資料載入與預處理

```python
class TrimodalDataset(Dataset):
    def __init__(self, mapping_file='features/trimodal_mapping.json'):
        with open(mapping_file, 'r') as f:
            self.trimodal_data = json.load(f)

        self.words = list(self.trimodal_data.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}

        # 準備所有檔案路徑
        self.samples = []
        for word, data in self.trimodal_data.items():
            visual_dir = Path(data['modalities']['visual']['mediapipe_dir'])
            visual_files = list(visual_dir.glob('*.npy'))

            for visual_file in visual_files:
                self.samples.append({
                    'word': word,
                    'label': self.word_to_idx[word],
                    'visual_file': visual_file,
                    'audio_file': data['modalities']['audio']['fusion_variants'][2],  # normalized
                    'text_idx': self.word_to_idx[word]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 載入視覺特徵
        visual_feat = np.load(sample['visual_file'])  # (100, 417)

        # 載入音訊特徵
        audio_feat = np.load(sample['audio_file'])  # (24,)

        # 載入文字特徵
        text_embeddings = np.load('features/text_embeddings/unified_embeddings.npy')
        text_feat = text_embeddings[sample['text_idx']]  # (300,)

        return {
            'visual': torch.FloatTensor(visual_feat),
            'audio': torch.FloatTensor(audio_feat),
            'text': torch.FloatTensor(text_feat),
            'label': torch.LongTensor([sample['label']]),
            'word': sample['word']
        }
```

### 3. 訓練配置

```python
# 超參數配置
config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 50,
    'hidden_dim': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'gradient_clip': 1.0,

    # 學習率調度
    'scheduler': 'cosine',
    'warmup_epochs': 5,

    # 模態權重
    'visual_weight': 1.0,
    'audio_weight': 1.0,
    'text_weight': 1.0,

    # 損失權重
    'classification_weight': 1.0,
    'contrastive_weight': 0.1,
    'regularization_weight': 0.01
}

# 最佳化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# 學習率調度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

## 模型評估

### 1. 評估指標

```python
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(visual, audio, text)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    # 計算指標
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')
    f1 = f1_score(ground_truths, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(ground_truths, predictions)
    }
```

### 2. 跨模態分析

```python
def cross_modal_analysis(model, test_loader):
    """分析不同模態組合的性能"""
    model.eval()
    results = {}

    # 單模態性能
    results['visual_only'] = evaluate_single_modality(model, test_loader, 'visual')
    results['audio_only'] = evaluate_single_modality(model, test_loader, 'audio')
    results['text_only'] = evaluate_single_modality(model, test_loader, 'text')

    # 雙模態性能
    results['visual_audio'] = evaluate_dual_modality(model, test_loader, 'visual', 'audio')
    results['visual_text'] = evaluate_dual_modality(model, test_loader, 'visual', 'text')
    results['audio_text'] = evaluate_dual_modality(model, test_loader, 'audio', 'text')

    # 三模態性能
    results['trimodal'] = evaluate_model(model, test_loader)

    return results
```

## 部署與應用

### 1. 模型匯出

```python
# 儲存完整模型
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'word_to_idx': dataset.word_to_idx,
    'idx_to_word': {idx: word for word, idx in dataset.word_to_idx.items()},
    'training_stats': training_stats
}, 'trimodal_sign_language_llm.pth')

# ONNX匯出（推理優化）
torch.onnx.export(
    model,
    (sample_visual, sample_audio, sample_text),
    'trimodal_model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['visual_input', 'audio_input', 'text_input'],
    output_names=['classification_output']
)
```

### 2. 即時推理介面

```python
class SignLanguageInference:
    def __init__(self, model_path, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        self.model = TrimodalSignLanguageLLM(**checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.idx_to_word = checkpoint['idx_to_word']
        self.device = device

    def predict(self, visual_seq, audio_feat, text_embed):
        """即時手語識別"""
        with torch.no_grad():
            visual_tensor = torch.FloatTensor(visual_seq).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor(audio_feat).unsqueeze(0).to(self.device)
            text_tensor = torch.FloatTensor(text_embed).unsqueeze(0).to(self.device)

            outputs = self.model(visual_tensor, audio_tensor, text_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][pred_idx].item()

            return {
                'predicted_word': self.idx_to_word[pred_idx],
                'confidence': confidence,
                'all_probs': probs[0].cpu().numpy()
            }
```

## 進階應用

### 1. 手語生成（條件生成）

```python
class SignLanguageGenerator(nn.Module):
    def __init__(self, trimodal_llm):
        super().__init__()
        self.encoder = trimodal_llm
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.visual_generator = nn.Linear(512, 417)  # 生成視覺序列

    def generate_sign_sequence(self, text_prompt, sequence_length=100):
        """根據文字提示生成手語動作序列"""
        # 文字編碼
        text_feat = self.encoder.text_encoder(text_prompt)

        # 序列生成
        generated_sequence = []
        hidden = text_feat.unsqueeze(0)

        for _ in range(sequence_length):
            output = self.decoder(hidden, hidden)
            visual_frame = self.visual_generator(output)
            generated_sequence.append(visual_frame)
            hidden = output

        return torch.stack(generated_sequence, dim=1)  # (1, 100, 417)
```

### 2. 多語言手語翻譯

```python
class MultilingualSignTranslator:
    def __init__(self, models_dict):
        self.models = models_dict  # {'en': model_en, 'zh': model_zh, ...}

    def translate_sign_to_multilingual(self, visual_seq, audio_feat):
        """將手語翻譯成多種語言"""
        results = {}

        for lang, model in self.models.items():
            # 使用對應語言的文字嵌入
            text_embed = self.get_language_embedding(lang)
            prediction = model.predict(visual_seq, audio_feat, text_embed)
            results[lang] = prediction

        return results
```

### 3. 個性化適應

```python
def personalized_adaptation(base_model, user_data, adaptation_epochs=10):
    """針對特定使用者的手語習慣進行模型微調"""

    # 凍結大部分參數，只微調注意力權重
    for param in base_model.parameters():
        param.requires_grad = False

    # 解凍跨模態注意力層
    for param in base_model.trimodal_fusion.parameters():
        param.requires_grad = True

    # 小學習率微調
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=1e-5
    )

    # 微調訓練
    for epoch in range(adaptation_epochs):
        for batch in user_data:
            loss = compute_adaptation_loss(base_model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return base_model
```

## 最佳實踐

### 1. 資料增強策略

```python
class MultimodalAugmentation:
    def __init__(self):
        self.visual_aug = VisualAugmentation()
        self.audio_aug = AudioAugmentation()

    def augment_sample(self, visual_seq, audio_feat, text_embed):
        """多模態資料增強"""

        # 視覺增強：時序抖動、關鍵點噪音
        aug_visual = self.visual_aug.temporal_jitter(visual_seq)
        aug_visual = self.visual_aug.add_keypoint_noise(aug_visual, noise_std=0.01)

        # 音訊增強：特徵混合、維度dropout
        aug_audio = self.audio_aug.feature_mixing(audio_feat)
        aug_audio = self.audio_aug.dimension_dropout(aug_audio, dropout_rate=0.1)

        # 文字增強：語義近義詞替換
        aug_text = self.text_aug.semantic_substitution(text_embed)

        return aug_visual, aug_audio, aug_text
```

### 2. 模型解釋性

```python
class ModelInterpretability:
    def __init__(self, model):
        self.model = model

    def attention_visualization(self, visual_seq, audio_feat, text_embed):
        """可視化跨模態注意力權重"""

        # 獲取注意力權重
        with torch.no_grad():
            visual_encoded = self.model.visual_encoder(visual_seq)
            audio_encoded = self.model.audio_encoder(audio_feat)
            text_encoded = self.model.text_encoder(text_embed)

            # 注意力權重
            va_attn = self.model.trimodal_fusion.visual_audio_attn
            vt_attn = self.model.trimodal_fusion.visual_text_attn
            at_attn = self.model.trimodal_fusion.audio_text_attn

        return {
            'visual_audio_attention': va_attn,
            'visual_text_attention': vt_attn,
            'audio_text_attention': at_attn,
            'modal_weights': self.model.trimodal_fusion.modal_weights
        }

    def feature_importance_analysis(self, sample):
        """分析不同特徵維度的重要性"""
        # 使用Integrated Gradients或LIME等技術
        pass
```

## 總結

本指南提供了完整的多模態手語識別LLM訓練方案，包括：

1. **完整的架構設計**：Trimodal Transformer with Cross-Modal Attention
2. **多階段訓練策略**：單模態預訓練 → 跨模態對齊 → 端到端聯合訓練
3. **豐富的評估方法**：單模態、雙模態、三模態性能對比
4. **實用的部署方案**：ONNX匯出、即時推理介面
5. **進階應用擴展**：手語生成、多語言翻譯、個性化適應

透過本專案的features資料夾，你可以實現一個真正的**端到端多模態手語理解系統**，不僅能進行準確的手語識別，還能支援跨模態檢索、手語生成等多種應用場景。

---

**下一步：** 根據本指南實作程式碼，建議先從單模態訓練開始，逐步構建完整的多模態融合系統。