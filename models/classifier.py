"""
Multi-Modal Sign Language Classifier
多模態手語分類器主模組
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .encoders import VisualEncoder, AudioEncoder, TextEncoder
from .fusion import ModalFusion, ContrastiveLoss


class MultiModalSignClassifier(nn.Module):
    """
    完整的多模態手語分類器

    特點:
    ✅ 靈活模態組合: 支援1-3個模態任意組合推理
    ✅ 統一特徵空間: 所有模態映射到512維
    ✅ 注意力機制: 自動學習模態重要性權重
    ✅ 端到端訓練: 從原始特徵到分類結果

    參數量估算:
    - Visual Encoder: ~3.8M
    - Audio Encoder: ~0.17M
    - Text Encoder: ~0.96M
    - Fusion Module: ~1.9M
    - Classifier: ~0.1M
    總計: ~7M parameters
    """

    def __init__(
        self,
        num_classes: int = 30,
        modalities: List[str] = ['visual', 'audio', 'text'],
        fusion_strategy: str = 'attention',
        text_embedding_type: str = 'unified',
        dropout: float = 0.1,
        use_contrastive_loss: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.modalities = modalities
        self.text_embedding_type = text_embedding_type
        self.use_contrastive_loss = use_contrastive_loss

        # 模態編碼器
        self.encoders = nn.ModuleDict()

        if 'visual' in modalities:
            self.encoders['visual'] = VisualEncoder(
                input_dim=417,
                hidden_dim=256,
                output_dim=512,
                dropout=dropout
            )

        if 'audio' in modalities:
            self.encoders['audio'] = AudioEncoder(
                input_dim=24,
                hidden_dims=[128, 256],
                output_dim=512,
                dropout=dropout
            )

        if 'text' in modalities:
            self.encoders['text'] = TextEncoder(
                embedding_type=text_embedding_type,
                output_dim=512,
                dropout=dropout
            )

        # 跨模態融合模組
        self.fusion = ModalFusion(
            feature_dim=512,
            fusion_strategy=fusion_strategy,
            dropout=dropout
        )

        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes)
        )

        # 對比學習損失
        if use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss()

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
        features: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        前向傳播

        Args:
            features: 特徵字典，可包含 'visual', 'audio', 'text'
            return_embeddings: 是否返回各模態編碼後的嵌入
            return_attention_weights: 是否返回注意力權重

        Returns:
            logits: 分類logits (batch, num_classes)
            embeddings: 各模態嵌入字典 (可選)
            attention_weights: 注意力權重字典 (可選)
        """
        # 1. 各模態編碼
        modal_embeddings = {}

        if 'visual' in features and 'visual' in self.encoders:
            modal_embeddings['visual'] = self.encoders['visual'](features['visual'])

        if 'audio' in features and 'audio' in self.encoders:
            modal_embeddings['audio'] = self.encoders['audio'](features['audio'])

        if 'text' in features and 'text' in self.encoders:
            modal_embeddings['text'] = self.encoders['text'](features['text'])

        # 2. 跨模態融合
        if return_attention_weights:
            fused_features, attention_weights = self.fusion(
                visual_features=modal_embeddings.get('visual'),
                audio_features=modal_embeddings.get('audio'),
                text_features=modal_embeddings.get('text'),
                return_attention_weights=True
            )
        else:
            fused_features = self.fusion(
                visual_features=modal_embeddings.get('visual'),
                audio_features=modal_embeddings.get('audio'),
                text_features=modal_embeddings.get('text')
            )

        # 3. 分類預測
        logits = self.classifier(fused_features)

        # 4. 構建返回值
        outputs = [logits]

        if return_embeddings:
            outputs.append(modal_embeddings)

        if return_attention_weights:
            outputs.append(attention_weights)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def compute_contrastive_loss(
        self,
        modal_embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        計算跨模態對比損失

        Args:
            modal_embeddings: 各模態嵌入字典
            labels: 標籤 (batch,)

        Returns:
            losses: 對比損失字典
        """
        if not self.use_contrastive_loss:
            return {}

        contrastive_losses = {}
        modality_names = list(modal_embeddings.keys())

        # 計算所有模態對的對比損失
        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                modal_a = modality_names[i]
                modal_b = modality_names[j]

                loss_key = f"{modal_a}_{modal_b}_contrastive"
                contrastive_losses[loss_key] = self.contrastive_loss(
                    modal_embeddings[modal_a],
                    modal_embeddings[modal_b],
                    labels
                )

        return contrastive_losses

    def set_modalities(self, modalities: List[str]):
        """動態設定使用的模態"""
        self.modalities = modalities

    def freeze_encoders(self, modalities: Optional[List[str]] = None):
        """凍結指定模態的編碼器參數"""
        if modalities is None:
            modalities = self.modalities

        for modality in modalities:
            if modality in self.encoders:
                for param in self.encoders[modality].parameters():
                    param.requires_grad = False

    def unfreeze_encoders(self, modalities: Optional[List[str]] = None):
        """解凍指定模態的編碼器參數"""
        if modalities is None:
            modalities = self.modalities

        for modality in modalities:
            if modality in self.encoders:
                for param in self.encoders[modality].parameters():
                    param.requires_grad = True

    def get_model_info(self) -> Dict[str, Union[int, str, List]]:
        """獲取模型資訊"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'num_classes': self.num_classes,
            'modalities': self.modalities,
            'text_embedding_type': self.text_embedding_type,
            'total_parameters': count_parameters(self),
            'encoder_parameters': {},
            'fusion_parameters': count_parameters(self.fusion),
            'classifier_parameters': count_parameters(self.classifier)
        }

        for modality, encoder in self.encoders.items():
            info['encoder_parameters'][modality] = count_parameters(encoder)

        return info


class EnsembleClassifier(nn.Module):
    """
    集成分類器
    結合多個不同配置的分類器進行預測
    """

    def __init__(self, classifiers: List[MultiModalSignClassifier], voting_strategy: str = 'soft'):
        super().__init__()

        self.classifiers = nn.ModuleList(classifiers)
        self.voting_strategy = voting_strategy
        self.num_classes = classifiers[0].num_classes

        # 驗證所有分類器的類別數一致
        for classifier in classifiers:
            assert classifier.num_classes == self.num_classes

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        集成前向傳播

        Args:
            features: 特徵字典

        Returns:
            ensemble_logits: 集成後的分類logits
        """
        predictions = []

        for classifier in self.classifiers:
            logits = classifier(features)
            if self.voting_strategy == 'soft':
                predictions.append(F.softmax(logits, dim=-1))
            else:  # hard voting
                predictions.append(logits)

        # 平均集成
        if self.voting_strategy == 'soft':
            ensemble_probs = torch.stack(predictions, dim=0).mean(dim=0)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)  # 轉回logits
        else:
            ensemble_logits = torch.stack(predictions, dim=0).mean(dim=0)

        return ensemble_logits


def create_multimodal_classifier(
    num_classes: int = 30,
    modalities: List[str] = ['visual', 'audio', 'text'],
    fusion_strategy: str = 'attention',
    text_embedding_type: str = 'unified',
    pretrained_encoders: Optional[Dict[str, str]] = None
) -> MultiModalSignClassifier:
    """
    創建多模態分類器的工廠函數

    Args:
        num_classes: 分類類別數
        modalities: 使用的模態列表
        fusion_strategy: 融合策略
        text_embedding_type: 文字嵌入類型
        pretrained_encoders: 預訓練編碼器權重路徑字典

    Returns:
        model: 多模態分類器實例
    """
    model = MultiModalSignClassifier(
        num_classes=num_classes,
        modalities=modalities,
        fusion_strategy=fusion_strategy,
        text_embedding_type=text_embedding_type
    )

    # 載入預訓練編碼器權重
    if pretrained_encoders:
        for modality, weight_path in pretrained_encoders.items():
            if modality in model.encoders:
                try:
                    state_dict = torch.load(weight_path, map_location='cpu')
                    model.encoders[modality].load_state_dict(state_dict)
                    print(f"✅ 載入 {modality} 編碼器預訓練權重: {weight_path}")
                except Exception as e:
                    print(f"⚠️  載入 {modality} 編碼器權重失敗: {e}")

    return model


def test_classifier():
    """測試完整分類器"""
    print("🧪 測試多模態分類器...")

    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建測試數據
    test_features = {
        'visual': torch.randn(batch_size, 100, 417).to(device),
        'audio': torch.randn(batch_size, 24).to(device),
        'text': torch.randn(batch_size, 300).to(device)
    }
    labels = torch.randint(0, 30, (batch_size,)).to(device)

    # 測試完整分類器
    print("\n1. 測試完整三模態分類器")
    model = MultiModalSignClassifier().to(device)

    # 基礎前向傳播
    logits = model(test_features)
    print(f"   輸出logits形狀: {logits.shape}")
    assert logits.shape == (batch_size, 30), "分類器輸出形狀錯誤"

    # 帶嵌入和注意力權重的前向傳播
    logits, embeddings, attention_weights = model(
        test_features,
        return_embeddings=True,
        return_attention_weights=True
    )
    print(f"   嵌入數量: {len(embeddings)}")
    print(f"   注意力權重數量: {len(attention_weights)}")

    # 測試對比損失
    contrastive_losses = model.compute_contrastive_loss(embeddings, labels)
    print(f"   對比損失數量: {len(contrastive_losses)}")

    # 測試不同模態組合
    print("\n2. 測試不同模態組合")
    for modalities in [['visual'], ['audio'], ['text'], ['visual', 'audio'], ['visual', 'text'], ['audio', 'text']]:
        subset_model = MultiModalSignClassifier(modalities=modalities).to(device)
        subset_features = {k: v for k, v in test_features.items() if k in modalities}
        subset_logits = subset_model(subset_features)
        print(f"   {'+'.join(modalities):>15} 模態: {subset_logits.shape}")

    # 測試模型資訊
    print("\n3. 模型資訊")
    model_info = model.get_model_info()
    print(f"   總參數數量: {model_info['total_parameters']:,}")
    print(f"   融合模組參數: {model_info['fusion_parameters']:,}")
    print(f"   分類器參數: {model_info['classifier_parameters']:,}")
    for modality, params in model_info['encoder_parameters'].items():
        print(f"   {modality} 編碼器參數: {params:,}")

    # 測試集成分類器
    print("\n4. 測試集成分類器")
    classifiers = [
        MultiModalSignClassifier(fusion_strategy='attention').to(device),
        MultiModalSignClassifier(fusion_strategy='concat').to(device),
        MultiModalSignClassifier(fusion_strategy='weighted_avg').to(device)
    ]
    ensemble = EnsembleClassifier(classifiers).to(device)
    ensemble_logits = ensemble(test_features)
    print(f"   集成輸出形狀: {ensemble_logits.shape}")

    print("\n✅ 分類器測試完成!")


if __name__ == "__main__":
    test_classifier()