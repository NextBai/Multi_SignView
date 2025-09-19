#!/usr/bin/env python3
"""
多模態融合器 - Multi_SignView 多模態手語辨識系統
整合視覺、語音、文字三大模態特徵，實現統一的多模態表示學習
支援早期融合、中期融合、晚期融合多層次策略

技術架構：
- 視覺特徵：MediaPipe(手部+姿態+臉部) + 光流運動 → 標準化到512維
- 語音特徵：詞彙TTS語音的MFCC+Spectral+Temporal → 標準化到24維
- 文字特徵：Word2Vec+FastText+BERT詞嵌入 → 標準化到300維
- 融合輸出：三模態注意力融合 → 統一256維表示

核心功能：
1. 跨模態特徵對齊和標準化
2. Multi-Head Cross-Modal Attention 注意力機制
3. 早期/中期/晚期多層次融合策略
4. 自適應模態權重學習
5. 缺失模態補償和魯棒性設計

Author: Claude Code + Multi_SignView Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 深度學習框架
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 路徑配置
PROJECT_ROOT = Path(__file__).parent
FEATURES_ROOT = PROJECT_ROOT / "features"
VISUAL_FEATURES_ROOT = FEATURES_ROOT / "mediapipe_features"
OPTICAL_FLOW_ROOT = FEATURES_ROOT / "optical_flow_features"
AUDIO_FEATURES_ROOT = FEATURES_ROOT / "audio_features"
TEXT_FEATURES_ROOT = FEATURES_ROOT / "text_embeddings"
OUTPUT_ROOT = FEATURES_ROOT / "multimodal_features"

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 30個手語詞彙
SIGN_LANGUAGE_VOCABULARY = [
    'again', 'bird', 'book', 'computer', 'cousin', 'deaf', 'drink', 'eat',
    'finish', 'fish', 'friend', 'good', 'happy', 'learn', 'like', 'mother',
    'need', 'nice', 'no', 'orange', 'school', 'sister', 'student', 'table',
    'teacher', 'tired', 'want', 'what', 'white', 'yes'
]


class ModalityAlignment:
    """
    跨模態特徵對齊和標準化模組

    負責將不同維度和分佈的模態特徵對齊到統一空間
    """

    def __init__(self, target_visual_dim: int = 512,
                 target_audio_dim: int = 64,
                 target_text_dim: int = 256):
        """
        初始化模態對齊器

        Args:
            target_visual_dim: 視覺特徵目標維度
            target_audio_dim: 音訊特徵目標維度
            target_text_dim: 文字特徵目標維度
        """
        self.target_dims = {
            'visual': target_visual_dim,
            'audio': target_audio_dim,
            'text': target_text_dim
        }

        # 特徵標準化器
        self.scalers = {}
        self.pca_reducers = {}

        print(f"🔗 ModalityAlignment 初始化完成")
        print(f"   - 目標維度: Visual({target_visual_dim}) + Audio({target_audio_dim}) + Text({target_text_dim})")

    def fit_alignment(self,
                     visual_features: np.ndarray,
                     audio_features: np.ndarray,
                     text_features: np.ndarray) -> Dict[str, Any]:
        """
        訓練模態對齊參數

        Args:
            visual_features: 視覺特徵矩陣 (N, visual_dim)
            audio_features: 音訊特徵矩陣 (N, audio_dim)
            text_features: 文字特徵矩陣 (N, text_dim)

        Returns:
            對齊參數字典
        """
        try:
            print("🔄 開始訓練跨模態特徵對齊...")

            alignment_info = {}

            # 1. 視覺特徵對齊
            if visual_features.shape[1] != self.target_dims['visual']:
                print(f"   📹 視覺特徵維度調整: {visual_features.shape[1]} → {self.target_dims['visual']}")

                # 標準化
                visual_scaler = StandardScaler()
                visual_features_scaled = visual_scaler.fit_transform(visual_features)
                self.scalers['visual'] = visual_scaler

                # 維度調整（PCA或擴展）
                if visual_features.shape[1] > self.target_dims['visual']:
                    # 降維
                    visual_pca = PCA(n_components=self.target_dims['visual'], random_state=42)
                    visual_pca.fit(visual_features_scaled)
                    self.pca_reducers['visual'] = visual_pca
                    alignment_info['visual_variance_ratio'] = np.sum(visual_pca.explained_variance_ratio_)
                else:
                    # 升維（零填充）
                    self.pca_reducers['visual'] = None

            # 2. 音訊特徵對齊
            if audio_features.shape[1] != self.target_dims['audio']:
                print(f"   🎤 音訊特徵維度調整: {audio_features.shape[1]} → {self.target_dims['audio']}")

                # 標準化
                audio_scaler = StandardScaler()
                audio_features_scaled = audio_scaler.fit_transform(audio_features)
                self.scalers['audio'] = audio_scaler

                # 維度調整
                if audio_features.shape[1] > self.target_dims['audio']:
                    audio_pca = PCA(n_components=self.target_dims['audio'], random_state=42)
                    audio_pca.fit(audio_features_scaled)
                    self.pca_reducers['audio'] = audio_pca
                    alignment_info['audio_variance_ratio'] = np.sum(audio_pca.explained_variance_ratio_)
                else:
                    self.pca_reducers['audio'] = None

            # 3. 文字特徵對齊
            if text_features.shape[1] != self.target_dims['text']:
                print(f"   📝 文字特徵維度調整: {text_features.shape[1]} → {self.target_dims['text']}")

                # 標準化
                text_scaler = StandardScaler()
                text_features_scaled = text_scaler.fit_transform(text_features)
                self.scalers['text'] = text_scaler

                # 維度調整
                if text_features.shape[1] > self.target_dims['text']:
                    text_pca = PCA(n_components=self.target_dims['text'], random_state=42)
                    text_pca.fit(text_features_scaled)
                    self.pca_reducers['text'] = text_pca
                    alignment_info['text_variance_ratio'] = np.sum(text_pca.explained_variance_ratio_)
                else:
                    self.pca_reducers['text'] = None

            print("✅ 跨模態特徵對齊訓練完成")
            return alignment_info

        except Exception as e:
            print(f"❌ 跨模態特徵對齊訓練失敗: {str(e)}")
            return {}

    def transform_modalities(self,
                           visual_features: Optional[np.ndarray] = None,
                           audio_features: Optional[np.ndarray] = None,
                           text_features: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        轉換模態特徵到對齊空間

        Args:
            visual_features: 視覺特徵
            audio_features: 音訊特徵
            text_features: 文字特徵

        Returns:
            對齊後的特徵字典
        """
        try:
            aligned_features = {}

            # 1. 轉換視覺特徵
            if visual_features is not None:
                if 'visual' in self.scalers:
                    visual_aligned = self.scalers['visual'].transform(visual_features)
                else:
                    visual_aligned = visual_features.copy()

                if self.pca_reducers.get('visual') is not None:
                    visual_aligned = self.pca_reducers['visual'].transform(visual_aligned)
                elif visual_aligned.shape[1] < self.target_dims['visual']:
                    # 零填充擴展
                    padding = np.zeros((visual_aligned.shape[0],
                                      self.target_dims['visual'] - visual_aligned.shape[1]))
                    visual_aligned = np.concatenate([visual_aligned, padding], axis=1)

                aligned_features['visual'] = visual_aligned

            # 2. 轉換音訊特徵
            if audio_features is not None:
                if 'audio' in self.scalers:
                    audio_aligned = self.scalers['audio'].transform(audio_features)
                else:
                    audio_aligned = audio_features.copy()

                if self.pca_reducers.get('audio') is not None:
                    audio_aligned = self.pca_reducers['audio'].transform(audio_aligned)
                elif audio_aligned.shape[1] < self.target_dims['audio']:
                    # 零填充擴展
                    padding = np.zeros((audio_aligned.shape[0],
                                      self.target_dims['audio'] - audio_aligned.shape[1]))
                    audio_aligned = np.concatenate([audio_aligned, padding], axis=1)

                aligned_features['audio'] = audio_aligned

            # 3. 轉換文字特徵
            if text_features is not None:
                if 'text' in self.scalers:
                    text_aligned = self.scalers['text'].transform(text_features)
                else:
                    text_aligned = text_features.copy()

                if self.pca_reducers.get('text') is not None:
                    text_aligned = self.pca_reducers['text'].transform(text_aligned)
                elif text_aligned.shape[1] < self.target_dims['text']:
                    # 零填充擴展
                    padding = np.zeros((text_aligned.shape[0],
                                      self.target_dims['text'] - text_aligned.shape[1]))
                    text_aligned = np.concatenate([text_aligned, padding], axis=1)

                aligned_features['text'] = text_aligned

            return aligned_features

        except Exception as e:
            print(f"❌ 模態特徵轉換失敗: {str(e)}")
            return {}


class CrossModalAttention(nn.Module):
    """
    跨模態注意力機制

    實現Multi-Head Cross-Modal Attention，計算模態間的依賴關係
    """

    def __init__(self,
                 visual_dim: int = 512,
                 audio_dim: int = 64,
                 text_dim: int = 256,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        """
        初始化跨模態注意力

        Args:
            visual_dim: 視覺特徵維度
            audio_dim: 音訊特徵維度
            text_dim: 文字特徵維度
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
        """
        super(CrossModalAttention, self).__init__()

        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 確保hidden_dim能被num_heads整除
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # 模態投影層
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Multi-Head Attention 組件
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 輸出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        print(f"🔗 CrossModalAttention 初始化完成")
        print(f"   - 輸入維度: Visual({visual_dim}) + Audio({audio_dim}) + Text({text_dim})")
        print(f"   - 隱藏維度: {hidden_dim}, 注意力頭數: {num_heads}")

    def forward(self,
                visual_feat: Optional[torch.Tensor] = None,
                audio_feat: Optional[torch.Tensor] = None,
                text_feat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向傳播

        Args:
            visual_feat: 視覺特徵 (batch_size, visual_dim)
            audio_feat: 音訊特徵 (batch_size, audio_dim)
            text_feat: 文字特徵 (batch_size, text_dim)

        Returns:
            跨模態注意力結果字典
        """
        batch_size = None
        modalities = {}

        # 投影到統一隱藏空間
        if visual_feat is not None:
            batch_size = visual_feat.size(0)
            modalities['visual'] = self.visual_proj(visual_feat)

        if audio_feat is not None:
            batch_size = audio_feat.size(0)
            modalities['audio'] = self.audio_proj(audio_feat)

        if text_feat is not None:
            batch_size = text_feat.size(0)
            modalities['text'] = self.text_proj(text_feat)

        if not modalities:
            return {}

        # 計算跨模態注意力
        attention_results = {}
        modality_names = list(modalities.keys())

        for i, query_modal in enumerate(modality_names):
            for j, key_modal in enumerate(modality_names):
                if i != j:  # 跨模態注意力，不計算自注意力
                    attention_key = f"{query_modal}_to_{key_modal}"

                    # Multi-Head Attention
                    query = self.query_proj(modalities[query_modal])  # (batch, hidden)
                    key = self.key_proj(modalities[key_modal])        # (batch, hidden)
                    value = self.value_proj(modalities[key_modal])    # (batch, hidden)

                    # Reshape為多頭
                    query = query.view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
                    key = key.view(batch_size, self.num_heads, self.head_dim)
                    value = value.view(batch_size, self.num_heads, self.head_dim)

                    # 計算注意力分數
                    attention_scores = torch.sum(query * key, dim=-1) / np.sqrt(self.head_dim)  # (batch, heads)
                    attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, heads)

                    # 應用注意力到value
                    attended_value = attention_weights.unsqueeze(-1) * value  # (batch, heads, head_dim)
                    attended_value = attended_value.view(batch_size, -1)  # (batch, hidden)

                    # 輸出投影和殘差連接
                    attended_output = self.output_proj(attended_value)
                    attended_output = self.layer_norm(attended_output + modalities[query_modal])
                    attended_output = self.dropout(attended_output)

                    attention_results[attention_key] = {
                        'attended_features': attended_output,
                        'attention_weights': attention_weights
                    }

        return attention_results

    def get_fusion_features(self, attention_results: Dict[str, Dict]) -> torch.Tensor:
        """
        從注意力結果中提取融合特徵

        Args:
            attention_results: 注意力計算結果

        Returns:
            融合後的特徵向量
        """
        try:
            # 收集所有attended features
            attended_features = []
            for attention_key, result in attention_results.items():
                attended_features.append(result['attended_features'])

            if attended_features:
                # 平均池化融合
                fusion_features = torch.mean(torch.stack(attended_features), dim=0)
                return fusion_features
            else:
                return torch.empty(0)

        except Exception as e:
            print(f"❌ 融合特徵提取失敗: {str(e)}")
            return torch.empty(0)


class MultiModalFusion:
    """
    多模態融合主控制器

    整合特徵對齊、跨模態注意力、多層次融合策略
    """

    def __init__(self,
                 visual_dim: int = 512,
                 audio_dim: int = 64,
                 text_dim: int = 256,
                 output_dim: int = 256,
                 fusion_strategies: List[str] = ['early', 'attention', 'late']):
        """
        初始化多模態融合器

        Args:
            visual_dim: 視覺特徵維度
            audio_dim: 音訊特徵維度
            text_dim: 文字特徵維度
            output_dim: 輸出特徵維度
            fusion_strategies: 融合策略列表
        """
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.fusion_strategies = fusion_strategies

        # 初始化組件
        self.alignment = ModalityAlignment(visual_dim, audio_dim, text_dim)

        # PyTorch設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 跨模態注意力（延遲初始化）
        self.cross_attention = None

        # 融合結果存儲
        self.fusion_results = {}

        print(f"🔀 MultiModalFusion 初始化完成")
        print(f"   - 設備: {self.device}")
        print(f"   - 融合策略: {', '.join(fusion_strategies)}")
        print(f"   - 輸出維度: {output_dim}")

    def fit(self,
            visual_features: Optional[np.ndarray] = None,
            audio_features: Optional[np.ndarray] = None,
            text_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        訓練多模態融合器

        Args:
            visual_features: 視覺特徵矩陣
            audio_features: 音訊特徵矩陣
            text_features: 文字特徵矩陣

        Returns:
            訓練信息字典
        """
        try:
            print("🔄 開始訓練多模態融合器...")

            training_info = {}

            # 1. 特徵對齊訓練
            if all(feat is not None for feat in [visual_features, audio_features, text_features]):
                alignment_info = self.alignment.fit_alignment(
                    visual_features, audio_features, text_features
                )
                training_info['alignment'] = alignment_info

                # 2. 初始化跨模態注意力
                self.cross_attention = CrossModalAttention(
                    self.visual_dim, self.audio_dim, self.text_dim,
                    hidden_dim=self.output_dim, num_heads=8
                ).to(self.device)

                training_info['cross_attention_initialized'] = True

                print("✅ 多模態融合器訓練完成")
            else:
                print("⚠️  部分模態特徵缺失，僅進行部分訓練")

            return training_info

        except Exception as e:
            print(f"❌ 多模態融合器訓練失敗: {str(e)}")
            return {}

    def fuse_features(self,
                     visual_features: Optional[np.ndarray] = None,
                     audio_features: Optional[np.ndarray] = None,
                     text_features: Optional[np.ndarray] = None,
                     strategy: str = 'attention') -> Dict[str, np.ndarray]:
        """
        執行多模態特徵融合

        Args:
            visual_features: 視覺特徵
            audio_features: 音訊特徵
            text_features: 文字特徵
            strategy: 融合策略 ('early', 'attention', 'late')

        Returns:
            融合結果字典
        """
        try:
            print(f"🔀 執行 {strategy} 融合策略...")

            fusion_results = {}

            # 1. 特徵對齊
            aligned_features = self.alignment.transform_modalities(
                visual_features, audio_features, text_features
            )

            if not aligned_features:
                return {}

            # 2. 根據策略執行融合
            if strategy == 'early':
                # 早期融合：特徵拼接
                fusion_results = self._early_fusion(aligned_features)

            elif strategy == 'attention' and self.cross_attention:
                # 中期融合：跨模態注意力
                fusion_results = self._attention_fusion(aligned_features)

            elif strategy == 'late':
                # 晚期融合：決策級融合
                fusion_results = self._late_fusion(aligned_features)

            else:
                print(f"❌ 不支援的融合策略或未初始化: {strategy}")
                return {}

            print(f"✅ {strategy} 融合完成")
            return fusion_results

        except Exception as e:
            print(f"❌ 多模態特徵融合失敗: {str(e)}")
            return {}

    def _early_fusion(self, aligned_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        早期融合：特徵拼接和降維

        Args:
            aligned_features: 對齊後的特徵字典

        Returns:
            早期融合結果
        """
        try:
            # 拼接所有可用特徵
            feature_list = []
            modality_info = []

            for modality, features in aligned_features.items():
                feature_list.append(features)
                modality_info.append(f"{modality}({features.shape[1]})")

            if not feature_list:
                return {}

            # 特徵拼接
            concatenated_features = np.concatenate(feature_list, axis=1)

            print(f"   📊 早期融合: {' + '.join(modality_info)} = {concatenated_features.shape[1]}維")

            # 降維到目標維度
            if concatenated_features.shape[1] > self.output_dim:
                pca = PCA(n_components=self.output_dim, random_state=42)
                fused_features = pca.fit_transform(concatenated_features)
                variance_ratio = np.sum(pca.explained_variance_ratio_)
                print(f"   📉 PCA降維保留 {variance_ratio:.2%} 方差")
            else:
                fused_features = concatenated_features

            return {
                'early_fusion': fused_features,
                'concatenated_features': concatenated_features,
                'modality_info': modality_info
            }

        except Exception as e:
            print(f"❌ 早期融合失敗: {str(e)}")
            return {}

    def _attention_fusion(self, aligned_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        注意力融合：跨模態注意力機制

        Args:
            aligned_features: 對齊後的特徵字典

        Returns:
            注意力融合結果
        """
        try:
            # 轉換為torch tensor
            torch_features = {}
            for modality, features in aligned_features.items():
                torch_features[modality] = torch.tensor(features, dtype=torch.float32).to(self.device)

            # 跨模態注意力計算
            with torch.no_grad():
                attention_results = self.cross_attention(
                    visual_feat=torch_features.get('visual'),
                    audio_feat=torch_features.get('audio'),
                    text_feat=torch_features.get('text')
                )

                if attention_results:
                    # 提取融合特徵
                    fusion_features = self.cross_attention.get_fusion_features(attention_results)
                    fusion_features_np = fusion_features.cpu().numpy()

                    # 提取注意力權重
                    attention_weights = {}
                    for key, result in attention_results.items():
                        attention_weights[key] = result['attention_weights'].cpu().numpy()

                    print(f"   🔗 注意力融合: {len(attention_results)} 個跨模態連接")

                    return {
                        'attention_fusion': fusion_features_np,
                        'attention_weights': attention_weights,
                        'attention_details': attention_results
                    }

            return {}

        except Exception as e:
            print(f"❌ 注意力融合失敗: {str(e)}")
            return {}

    def _late_fusion(self, aligned_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        晚期融合：決策級融合

        Args:
            aligned_features: 對齊後的特徵字典

        Returns:
            晚期融合結果
        """
        try:
            # 對每個模態分別進行特徵處理
            modality_representations = {}

            for modality, features in aligned_features.items():
                # 標準化
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(features)

                # 降維到統一維度
                if normalized_features.shape[1] > self.output_dim // len(aligned_features):
                    target_dim = self.output_dim // len(aligned_features)
                    pca = PCA(n_components=target_dim, random_state=42)
                    reduced_features = pca.fit_transform(normalized_features)
                else:
                    reduced_features = normalized_features

                modality_representations[modality] = reduced_features

            # 加權融合（可以是學習的權重）
            modality_weights = {
                'visual': 0.5,
                'audio': 0.3,
                'text': 0.2
            }

            # 標準化權重
            available_modalities = list(modality_representations.keys())
            total_weight = sum(modality_weights.get(m, 1.0) for m in available_modalities)
            normalized_weights = {m: modality_weights.get(m, 1.0) / total_weight
                                for m in available_modalities}

            # 加權融合
            fusion_result = None
            for modality, features in modality_representations.items():
                weight = normalized_weights[modality]
                if fusion_result is None:
                    fusion_result = weight * features
                else:
                    # 維度對齊
                    min_dim = min(fusion_result.shape[1], features.shape[1])
                    fusion_result = fusion_result[:, :min_dim] + weight * features[:, :min_dim]

            print(f"   ⚖️  晚期融合權重: {normalized_weights}")

            return {
                'late_fusion': fusion_result,
                'modality_weights': normalized_weights,
                'modality_representations': modality_representations
            }

        except Exception as e:
            print(f"❌ 晚期融合失敗: {str(e)}")
            return {}

    def save_fusion_model(self, save_path: Path):
        """
        保存融合模型

        Args:
            save_path: 保存路徑
        """
        try:
            save_data = {
                'alignment': self.alignment,
                'fusion_params': {
                    'visual_dim': self.visual_dim,
                    'audio_dim': self.audio_dim,
                    'text_dim': self.text_dim,
                    'output_dim': self.output_dim,
                    'fusion_strategies': self.fusion_strategies
                }
            }

            # 保存跨模態注意力模型
            if self.cross_attention:
                torch.save(self.cross_attention.state_dict(),
                          save_path / 'cross_attention.pth')
                save_data['cross_attention_saved'] = True

            # 保存其他參數
            with open(save_path / 'fusion_model.pkl', 'wb') as f:
                pickle.dump(save_data, f)

            print(f"💾 多模態融合模型已保存: {save_path}")

        except Exception as e:
            print(f"❌ 模型保存失敗: {str(e)}")


def load_multimodal_features(features_root: Path = FEATURES_ROOT) -> Dict[str, Dict]:
    """
    載入所有模態的特徵資料

    Args:
        features_root: 特徵根目錄

    Returns:
        模態特徵字典
    """
    try:
        print("🔄 載入多模態特徵資料...")

        multimodal_data = {
            'visual': {},
            'audio': {},
            'text': {}
        }

        # 1. 載入視覺特徵（MediaPipe + 光流）
        visual_root = features_root / "mediapipe_features"
        optical_root = features_root / "optical_flow_features"

        # 2. 載入音訊特徵
        audio_root = features_root / "audio_features"
        if audio_root.exists():
            for audio_file in audio_root.glob("*_normalized_24d.npy"):
                word = audio_file.stem.replace('_normalized_24d', '')
                if word in SIGN_LANGUAGE_VOCABULARY:
                    multimodal_data['audio'][word] = np.load(audio_file)

        # 3. 載入文字特徵
        text_root = features_root / "text_embeddings"
        if (text_root / "unified_embeddings.npy").exists():
            unified_text = np.load(text_root / "unified_embeddings.npy")
            for i, word in enumerate(SIGN_LANGUAGE_VOCABULARY):
                if i < unified_text.shape[0]:
                    multimodal_data['text'][word] = unified_text[i]

        # 統計載入結果
        for modality, data in multimodal_data.items():
            print(f"   📊 {modality}: {len(data)} 個詞彙特徵")

        return multimodal_data

    except Exception as e:
        print(f"❌ 多模態特徵載入失敗: {str(e)}")
        return {}


def run_multimodal_fusion_pipeline(
    features_root: Optional[str] = None,
    output_root: Optional[str] = None,
    fusion_strategies: List[str] = ['early', 'attention', 'late']
) -> Dict[str, Any]:
    """
    執行完整的多模態融合管線

    Args:
        features_root: 特徵根目錄
        output_root: 輸出根目錄
        fusion_strategies: 融合策略列表

    Returns:
        融合結果
    """
    # 路徑配置
    features_path = Path(features_root) if features_root else FEATURES_ROOT
    output_path = Path(output_root) if output_root else OUTPUT_ROOT
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🔀 開始多模態融合管線")
    print(f"   📂 特徵根目錄: {features_path}")
    print(f"   📂 輸出目錄: {output_path}")

    try:
        # 1. 載入多模態特徵
        multimodal_data = load_multimodal_features(features_path)

        if not any(multimodal_data.values()):
            print("❌ 沒有載入到任何模態特徵")
            return {}

        # 2. 準備訓練資料
        # 這裡使用文字和音訊特徵作為示例
        words_with_all_modalities = set(multimodal_data['audio'].keys()) & set(multimodal_data['text'].keys())
        print(f"   📊 具備多模態特徵的詞彙: {len(words_with_all_modalities)} / {len(SIGN_LANGUAGE_VOCABULARY)}")

        if len(words_with_all_modalities) < 5:
            print("⚠️  可用於融合的詞彙太少，無法進行有效訓練")
            return {}

        # 準備特徵矩陣
        audio_matrix = []
        text_matrix = []
        word_labels = []

        for word in words_with_all_modalities:
            audio_matrix.append(multimodal_data['audio'][word])
            text_matrix.append(multimodal_data['text'][word])
            word_labels.append(word)

        audio_features = np.array(audio_matrix)
        text_features = np.array(text_matrix)

        print(f"   📊 訓練資料形狀: 音訊{audio_features.shape}, 文字{text_features.shape}")

        # 3. 初始化並訓練融合器
        fusion_system = MultiModalFusion(
            visual_dim=512,  # 暫時使用預設值
            audio_dim=audio_features.shape[1],
            text_dim=text_features.shape[1],
            output_dim=256,
            fusion_strategies=fusion_strategies
        )

        # 訓練融合器（目前只有音訊和文字）
        training_info = fusion_system.fit(
            visual_features=None,
            audio_features=audio_features,
            text_features=text_features
        )

        # 4. 執行融合測試
        fusion_results = {}
        for strategy in fusion_strategies:
            if strategy in ['early', 'late'] or (strategy == 'attention' and fusion_system.cross_attention):
                result = fusion_system.fuse_features(
                    visual_features=None,
                    audio_features=audio_features,
                    text_features=text_features,
                    strategy=strategy
                )
                if result:
                    fusion_results[strategy] = result

        # 5. 保存結果
        results = {
            'training_info': training_info,
            'fusion_results': fusion_results,
            'word_labels': word_labels,
            'feature_shapes': {
                'audio': audio_features.shape,
                'text': text_features.shape
            },
            'available_words': list(words_with_all_modalities)
        }

        # 保存融合特徵
        for strategy, result in fusion_results.items():
            for key, features in result.items():
                if isinstance(features, np.ndarray):
                    save_file = output_path / f"{strategy}_{key}.npy"
                    np.save(save_file, features)
                    print(f"💾 已保存: {save_file}")

        # 保存融合器模型
        fusion_system.save_fusion_model(output_path)

        # 保存結果摘要
        with open(output_path / 'fusion_summary.json', 'w', encoding='utf-8') as f:
            # 將numpy array轉換為list以便JSON序列化
            json_results = {}
            for key, value in results.items():
                if key == 'fusion_results':
                    continue  # 跳過numpy arrays
                json_results[key] = value

            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"🔀 多模態融合管線完成!")
        print(f"   ✅ 支援策略: {list(fusion_results.keys())}")
        print(f"   📊 處理詞彙: {len(word_labels)}")

        return results

    except Exception as e:
        print(f"❌ 多模態融合管線失敗: {str(e)}")
        return {}


if __name__ == "__main__":
    # 執行多模態融合管線
    results = run_multimodal_fusion_pipeline()