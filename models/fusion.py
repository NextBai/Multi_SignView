"""
Cross-Modal Fusion Mechanisms
跨模態融合機制實作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class CrossModalAttention(nn.Module):
    """
    跨模態注意力機制
    計算不同模態間的交互注意力

    支援的模態對:
    - Visual ↔ Audio: 動作與發音的時序對應
    - Visual ↔ Text: 手勢與語義的概念映射
    - Audio ↔ Text: 發音與詞彙的語音對齊
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.temperature = temperature

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # 多頭注意力機制
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 層正規化
        self.layer_norm = nn.LayerNorm(feature_dim)

        # 位置編碼 (用於序列模態)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, feature_dim) * 0.1)

    def forward(
        self,
        query_modal: torch.Tensor,
        key_value_modal: torch.Tensor,
        is_query_sequence: bool = False,
        is_kv_sequence: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算跨模態注意力

        Args:
            query_modal: 查詢模態特徵 (batch, [seq_len,] feature_dim)
            key_value_modal: 鍵值模態特徵 (batch, [seq_len,] feature_dim)
            is_query_sequence: 查詢模態是否為序列
            is_kv_sequence: 鍵值模態是否為序列

        Returns:
            attended_features: 注意力加權後的特徵
            attention_weights: 注意力權重
        """
        batch_size = query_modal.size(0)

        # 處理非序列數據，擴展為序列
        if not is_query_sequence:
            query_modal = query_modal.unsqueeze(1)  # (batch, 1, feature_dim)
        if not is_kv_sequence:
            key_value_modal = key_value_modal.unsqueeze(1)  # (batch, 1, feature_dim)

        # 添加位置編碼 (僅對真正的序列數據)
        if is_query_sequence and query_modal.size(1) <= 100:
            seq_len = query_modal.size(1)
            query_modal = query_modal + self.pos_encoding[:, :seq_len, :]

        if is_kv_sequence and key_value_modal.size(1) <= 100:
            seq_len = key_value_modal.size(1)
            key_value_modal = key_value_modal + self.pos_encoding[:, :seq_len, :]

        # 多頭注意力計算
        attended_features, attention_weights = self.attention(
            query=query_modal,
            key=key_value_modal,
            value=key_value_modal
        )

        # 層正規化和殘差連接
        attended_features = self.layer_norm(attended_features + query_modal)

        # 如果原始輸入不是序列，則壓縮回原始形狀
        if not is_query_sequence:
            attended_features = attended_features.squeeze(1)
            attention_weights = attention_weights.squeeze(1)

        return attended_features, attention_weights


class AdaptiveModalWeighting(nn.Module):
    """
    自適應模態權重學習
    根據輸入動態調整不同模態的重要性
    """

    def __init__(
        self,
        feature_dim: int = 512,
        max_modalities: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.max_modalities = max_modalities

        # 為不同數量的模態創建權重生成網路
        self.weight_generators = nn.ModuleDict()
        for num_modals in range(1, max_modalities + 1):
            self.weight_generators[str(num_modals)] = nn.Sequential(
                nn.Linear(feature_dim * num_modals, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_modals),
                nn.Softmax(dim=-1)
            )

    def forward(self, modal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        計算自適應權重

        Args:
            modal_features: 模態特徵列表

        Returns:
            weights: 自適應權重 (batch, num_modalities)
        """
        num_modalities = len(modal_features)

        if num_modalities == 1:
            # 單模態情況，權重為1
            batch_size = modal_features[0].size(0)
            return torch.ones(batch_size, 1, device=modal_features[0].device)

        # 拼接所有模態特徵
        concatenated = torch.cat(modal_features, dim=-1)

        # 根據模態數量選擇對應的權重生成器
        generator_key = str(num_modalities)
        if generator_key not in self.weight_generators:
            raise ValueError(f"不支援{num_modalities}個模態的權重生成")

        # 生成權重
        weights = self.weight_generators[generator_key](concatenated)

        return weights


class ModalFusion(nn.Module):
    """
    多模態融合模組
    支援多種融合策略和靈活的模態組合
    """

    def __init__(
        self,
        feature_dim: int = 512,
        fusion_strategy: str = 'attention',  # 'attention', 'concat', 'weighted_avg'
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_weighting: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy
        self.use_adaptive_weighting = use_adaptive_weighting

        # 跨模態注意力模組
        self.cross_attention = CrossModalAttention(
            feature_dim=feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # 自適應權重模組
        if use_adaptive_weighting:
            self.adaptive_weighting = AdaptiveModalWeighting(feature_dim)

        # 融合後的投影層
        if fusion_strategy == 'concat':
            # 拼接融合需要降維
            self.fusion_projection = nn.Sequential(
                nn.Linear(feature_dim * 3, feature_dim),  # 假設最多3個模態
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            # 其他策略保持原維度
            self.fusion_projection = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout)
            )

        # 模態特定的注意力權重 (可學習參數)
        self.modal_attention_weights = nn.Parameter(
            torch.ones(3, 3) / 3  # 3×3 模態對注意力權重矩陣
        )

    def forward(
        self,
        visual_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        多模態融合前向傳播

        Args:
            visual_features: 視覺特徵 (batch, feature_dim)
            audio_features: 音訊特徵 (batch, feature_dim)
            text_features: 文字特徵 (batch, feature_dim)
            return_attention_weights: 是否返回注意力權重

        Returns:
            fused_features: 融合後的特徵 (batch, feature_dim)
            attention_weights: 注意力權重字典 (可選)
        """
        available_modalities = []
        modal_features = []

        # 收集可用的模態
        if visual_features is not None:
            available_modalities.append('visual')
            modal_features.append(visual_features)

        if audio_features is not None:
            available_modalities.append('audio')
            modal_features.append(audio_features)

        if text_features is not None:
            available_modalities.append('text')
            modal_features.append(text_features)

        if len(modal_features) == 0:
            raise ValueError("至少需要一個模態的特徵")

        # 單模態情況：直接返回
        if len(modal_features) == 1:
            fused = self.fusion_projection(modal_features[0])
            if return_attention_weights:
                return fused, {}
            return fused

        # 多模態融合
        fused_features, attention_weights = self._fuse_multiple_modalities(
            modal_features, available_modalities
        )

        if return_attention_weights:
            return fused_features, attention_weights
        return fused_features

    def _fuse_multiple_modalities(
        self,
        modal_features: List[torch.Tensor],
        modality_names: List[str]
    ) -> Tuple[torch.Tensor, Dict]:
        """融合多個模態"""
        attention_weights = {}

        if self.fusion_strategy == 'attention':
            # 注意力融合策略
            fused_features = self._attention_fusion(modal_features, modality_names, attention_weights)

        elif self.fusion_strategy == 'concat':
            # 拼接融合策略
            fused_features = self._concat_fusion(modal_features)

        elif self.fusion_strategy == 'weighted_avg':
            # 加權平均融合策略
            fused_features = self._weighted_avg_fusion(modal_features)

        else:
            raise ValueError(f"不支援的融合策略: {self.fusion_strategy}")

        return fused_features, attention_weights

    def _attention_fusion(
        self,
        modal_features: List[torch.Tensor],
        modality_names: List[str],
        attention_weights: Dict
    ) -> torch.Tensor:
        """注意力融合實作"""
        num_modalities = len(modal_features)
        batch_size = modal_features[0].size(0)

        # 計算所有模態對的交互注意力
        attended_features = []
        for i in range(num_modalities):
            query_modal = modal_features[i]
            attended_sum = torch.zeros_like(query_modal)

            for j in range(num_modalities):
                if i != j:
                    key_value_modal = modal_features[j]

                    # 計算跨模態注意力
                    attended, attn_weights = self.cross_attention(
                        query_modal=query_modal,
                        key_value_modal=key_value_modal,
                        is_query_sequence=False,
                        is_kv_sequence=False
                    )

                    # 記錄注意力權重
                    pair_name = f"{modality_names[i]}_{modality_names[j]}"
                    attention_weights[pair_name] = attn_weights

                    # 應用可學習的模態對權重
                    modal_weight = self.modal_attention_weights[i, j]
                    attended_sum += modal_weight * attended

            attended_features.append(attended_sum)

        # 自適應權重融合
        if self.use_adaptive_weighting and len(attended_features) > 1:
            adaptive_weights = self.adaptive_weighting(attended_features)

            # 加權平均
            fused = torch.zeros_like(attended_features[0])
            for i, features in enumerate(attended_features):
                fused += adaptive_weights[:, i:i+1] * features
        else:
            # 簡單平均
            fused = torch.stack(attended_features, dim=0).mean(dim=0)

        return self.fusion_projection(fused)

    def _concat_fusion(self, modal_features: List[torch.Tensor]) -> torch.Tensor:
        """拼接融合實作"""
        # 簡單拼接所有模態特徵
        concatenated = torch.cat(modal_features, dim=-1)
        return self.fusion_projection(concatenated)

    def _weighted_avg_fusion(self, modal_features: List[torch.Tensor]) -> torch.Tensor:
        """加權平均融合實作"""
        if self.use_adaptive_weighting:
            # 使用自適應權重
            adaptive_weights = self.adaptive_weighting(modal_features)

            fused = torch.zeros_like(modal_features[0])
            for i, features in enumerate(modal_features):
                fused += adaptive_weights[:, i:i+1] * features
        else:
            # 簡單平均
            fused = torch.stack(modal_features, dim=0).mean(dim=0)

        return self.fusion_projection(fused)


class ContrastiveLoss(nn.Module):
    """
    跨模態對比學習損失
    促進同類樣本的跨模態特徵對齊
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        計算對比損失

        Args:
            features_a: 模態A的特徵 (batch, feature_dim)
            features_b: 模態B的特徵 (batch, feature_dim)
            labels: 標籤 (batch,)

        Returns:
            loss: 對比損失值
        """
        # L2正規化
        features_a = F.normalize(features_a, p=2, dim=1)
        features_b = F.normalize(features_b, p=2, dim=1)

        # 計算相似度矩陣
        similarity_matrix = torch.matmul(features_a, features_b.T) / self.temperature

        # 創建正樣本遮罩 (同類別為正樣本)
        labels_expanded_a = labels.unsqueeze(1).expand(-1, labels.size(0))
        labels_expanded_b = labels.unsqueeze(0).expand(labels.size(0), -1)
        positive_mask = (labels_expanded_a == labels_expanded_b).float()

        # 對角線遮罩 (排除自己與自己的相似度)
        eye_mask = torch.eye(labels.size(0), device=labels.device)
        positive_mask = positive_mask * (1 - eye_mask)

        # 計算InfoNCE損失
        exp_sim = torch.exp(similarity_matrix)

        # 正樣本的對數似然
        positive_sum = (positive_mask * exp_sim).sum(dim=1)

        # 所有樣本的對數似然
        all_sum = exp_sim.sum(dim=1)

        # 避免除零
        positive_sum = torch.clamp(positive_sum, min=1e-8)

        # 計算損失
        loss = -torch.log(positive_sum / all_sum).mean()

        return loss


def test_fusion_mechanisms():
    """測試融合機制"""
    print("🧪 測試跨模態融合機制...")

    batch_size = 4
    feature_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建測試數據
    visual_feat = torch.randn(batch_size, feature_dim).to(device)
    audio_feat = torch.randn(batch_size, feature_dim).to(device)
    text_feat = torch.randn(batch_size, feature_dim).to(device)
    labels = torch.randint(0, 30, (batch_size,)).to(device)

    # 測試跨模態注意力
    print("\n1. 測試跨模態注意力")
    cross_attention = CrossModalAttention().to(device)
    attended_feat, attn_weights = cross_attention(visual_feat, audio_feat)
    print(f"   輸入形狀: {visual_feat.shape}, {audio_feat.shape}")
    print(f"   輸出形狀: {attended_feat.shape}")
    print(f"   注意力權重形狀: {attn_weights.shape}")

    # 測試模態融合
    print("\n2. 測試模態融合")
    fusion_module = ModalFusion(fusion_strategy='attention').to(device)

    # 測試三模態融合
    fused_feat, attention_dict = fusion_module(
        visual_features=visual_feat,
        audio_features=audio_feat,
        text_features=text_feat,
        return_attention_weights=True
    )
    print(f"   三模態融合輸出: {fused_feat.shape}")
    print(f"   注意力權重數量: {len(attention_dict)}")

    # 測試雙模態融合
    dual_fused = fusion_module(visual_features=visual_feat, audio_features=audio_feat)
    print(f"   雙模態融合輸出: {dual_fused.shape}")

    # 測試單模態
    single_fused = fusion_module(visual_features=visual_feat)
    print(f"   單模態融合輸出: {single_fused.shape}")

    # 測試對比損失
    print("\n3. 測試對比損失")
    contrastive_loss = ContrastiveLoss().to(device)
    loss_value = contrastive_loss(visual_feat, audio_feat, labels)
    print(f"   對比損失值: {loss_value.item():.4f}")

    # 測試不同融合策略
    print("\n4. 測試不同融合策略")
    for strategy in ['attention', 'concat', 'weighted_avg']:
        fusion = ModalFusion(fusion_strategy=strategy).to(device)
        output = fusion(visual_feat, audio_feat, text_feat)
        print(f"   {strategy:>12} 策略輸出: {output.shape}")

    print("\n✅ 融合機制測試完成!")

    # 計算參數量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 融合模組參數統計:")
    print(f"   跨模態注意力: {count_parameters(cross_attention):,} 參數")
    print(f"   模態融合模組: {count_parameters(fusion_module):,} 參數")


if __name__ == "__main__":
    test_fusion_mechanisms()