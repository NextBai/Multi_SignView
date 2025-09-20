"""
Multi-Modal Encoders for Sign Language Recognition
多模態編碼器實作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    位置編碼模組
    為序列數據添加位置資訊
    """

    def __init__(self, d_model: int, max_seq_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 創建位置編碼矩陣
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VisualEncoder(nn.Module):
    """
    視覺編碼器
    處理手語視覺序列特徵 (MediaPipe landmarks)

    架構:
    Input: (batch, 100, 417) - 100幀的417維MediaPipe特徵
    ↓
    1D Conv: 時序特徵提取
    ↓
    Transformer: 長期依賴建模
    ↓
    Global Pool: 全域特徵聚合
    ↓
    Output: (batch, 512) - 統一視覺表示
    """

    def __init__(
        self,
        input_dim: int = 417,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel_size: int = 5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 1. 時序卷積層 - 提取局部時序模式
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 2. 位置編碼
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len=100, dropout=dropout)

        # 3. Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False  # (seq_len, batch, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 4. 全域池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 5. 輸出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 視覺特徵序列 (batch, seq_len, feature_dim)
            mask: 可選的注意力遮罩 (batch, seq_len)

        Returns:
            encoded_features: 編碼後的特徵 (batch, output_dim)
        """
        batch_size, seq_len, feature_dim = x.shape

        # 1. 時序卷積 - 需要 (batch, feature, seq_len) 格式
        x_conv = x.transpose(1, 2)  # (batch, feature_dim, seq_len)
        x_conv = self.temporal_conv(x_conv)  # (batch, hidden_dim, seq_len)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # 2. 準備Transformer輸入 - 需要 (seq_len, batch, feature) 格式
        x_transformer = x_conv.transpose(0, 1)  # (seq_len, batch, hidden_dim)

        # 3. 添加位置編碼
        x_transformer = self.pos_encoding(x_transformer)

        # 4. Transformer編碼
        if mask is not None:
            # 創建注意力遮罩 (可選)
            attention_mask = self._create_attention_mask(mask)
            encoded = self.transformer(x_transformer, src_key_padding_mask=attention_mask)
        else:
            encoded = self.transformer(x_transformer)

        # 5. 轉換回 (batch, seq_len, feature) 格式
        encoded = encoded.transpose(0, 1)  # (batch, seq_len, hidden_dim)

        # 6. 全域池化 - 需要 (batch, feature, seq_len) 格式
        pooled = encoded.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        pooled = self.global_pool(pooled).squeeze(-1)  # (batch, hidden_dim)

        # 7. 輸出投影
        output = self.output_projection(pooled)  # (batch, output_dim)

        return output

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """創建Transformer的注意力遮罩"""
        # mask: (batch, seq_len) - True為有效位置
        # 返回: (batch, seq_len) - True為需要遮蔽的位置
        return ~mask


class AudioEncoder(nn.Module):
    """
    音訊編碼器
    處理音訊特徵 (MFCC + Spectral + Temporal)

    架構:
    Input: (batch, 24) - 24維音訊特徵
    ↓
    Multi-layer MLP with BatchNorm and Dropout
    ↓
    Output: (batch, 512) - 統一音訊表示
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: list = [128, 256],
        output_dim: int = 512,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 構建多層感知器
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 音訊特徵 (batch, input_dim)

        Returns:
            encoded_features: 編碼後的特徵 (batch, output_dim)
        """
        return self.encoder(x)


class TextEncoder(nn.Module):
    """
    文字編碼器
    處理文字嵌入特徵 (Word2Vec/FastText/BERT/Unified)

    架構:
    Input: (batch, embedding_dim) - 詞嵌入特徵
    ↓
    Layer Normalization + MLP
    ↓
    Output: (batch, 512) - 統一文字表示
    """

    def __init__(
        self,
        embedding_dims: dict = {
            'unified': 300,
            'word2vec': 300,
            'fasttext': 300,
            'bert': 768
        },
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        embedding_type: str = 'unified'
    ):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.output_dim = output_dim
        self.embedding_type = embedding_type

        # 為每種嵌入類型創建投影層
        self.projections = nn.ModuleDict()
        for emb_type, emb_dim in embedding_dims.items():
            self.projections[emb_type] = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        embedding_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 文字嵌入特徵 (batch, embedding_dim)
            embedding_type: 指定嵌入類型，如果為None則使用預設類型

        Returns:
            encoded_features: 編碼後的特徵 (batch, output_dim)
        """
        if embedding_type is None:
            embedding_type = self.embedding_type

        if embedding_type not in self.projections:
            raise ValueError(f"不支援的嵌入類型: {embedding_type}")

        return self.projections[embedding_type](x)

    def set_embedding_type(self, embedding_type: str):
        """設定預設嵌入類型"""
        if embedding_type not in self.embedding_dims:
            raise ValueError(f"不支援的嵌入類型: {embedding_type}")
        self.embedding_type = embedding_type


class AdaptivePooling1D(nn.Module):
    """
    自適應池化層
    可以處理不同長度的序列
    """

    def __init__(self, output_size: int = 1, pooling_type: str = 'avg'):
        super().__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type

        if pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(output_size)
        else:
            raise ValueError(f"不支援的池化類型: {pooling_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature, seq_len)
        Returns:
            pooled: (batch, feature, output_size)
        """
        return self.pool(x)


def test_encoders():
    """測試編碼器功能"""
    print("🧪 測試編碼器模組...")

    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試視覺編碼器
    print("\n1. 測試視覺編碼器")
    visual_encoder = VisualEncoder().to(device)
    visual_input = torch.randn(batch_size, 100, 417).to(device)

    visual_output = visual_encoder(visual_input)
    print(f"   輸入形狀: {visual_input.shape}")
    print(f"   輸出形狀: {visual_output.shape}")
    assert visual_output.shape == (batch_size, 512), "視覺編碼器輸出形狀錯誤"

    # 測試音訊編碼器
    print("\n2. 測試音訊編碼器")
    audio_encoder = AudioEncoder().to(device)
    audio_input = torch.randn(batch_size, 24).to(device)

    audio_output = audio_encoder(audio_input)
    print(f"   輸入形狀: {audio_input.shape}")
    print(f"   輸出形狀: {audio_output.shape}")
    assert audio_output.shape == (batch_size, 512), "音訊編碼器輸出形狀錯誤"

    # 測試文字編碼器
    print("\n3. 測試文字編碼器")
    text_encoder = TextEncoder().to(device)
    text_input = torch.randn(batch_size, 300).to(device)  # unified嵌入

    text_output = text_encoder(text_input)
    print(f"   輸入形狀: {text_input.shape}")
    print(f"   輸出形狀: {text_output.shape}")
    assert text_output.shape == (batch_size, 512), "文字編碼器輸出形狀錯誤"

    # 測試不同嵌入類型
    bert_input = torch.randn(batch_size, 768).to(device)
    bert_output = text_encoder(bert_input, embedding_type='bert')
    print(f"   BERT輸入形狀: {bert_input.shape}")
    print(f"   BERT輸出形狀: {bert_output.shape}")
    assert bert_output.shape == (batch_size, 512), "BERT編碼器輸出形狀錯誤"

    print("\n✅ 所有編碼器測試通過!")

    # 計算參數量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 模型參數統計:")
    print(f"   視覺編碼器: {count_parameters(visual_encoder):,} 參數")
    print(f"   音訊編碼器: {count_parameters(audio_encoder):,} 參數")
    print(f"   文字編碼器: {count_parameters(text_encoder):,} 參數")
    total_params = count_parameters(visual_encoder) + count_parameters(audio_encoder) + count_parameters(text_encoder)
    print(f"   總計: {total_params:,} 參數")


if __name__ == "__main__":
    test_encoders()