"""
Data Loaders for Multi-Modal Sign Language Dataset
多模態手語數據載入器
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .dataset import TriModalDataset, SemanticBalancedSampler


def collate_multimodal_batch(batch: List[Tuple[Dict[str, torch.Tensor], int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    多模態批次整理函數

    Args:
        batch: 批次數據列表，每個元素為 (features_dict, label)

    Returns:
        batched_features: 批次特徵字典
        batched_labels: 批次標籤張量
    """
    # 分離特徵和標籤
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # 獲取所有可能的模態
    all_modalities = set()
    for features in features_list:
        all_modalities.update(features.keys())

    # 為每個模態構建批次張量
    batched_features = {}

    for modality in all_modalities:
        # 收集該模態的所有特徵
        modality_features = []
        for features in features_list:
            if modality in features:
                modality_features.append(features[modality])
            else:
                # 如果某個樣本缺少該模態，使用零張量填充
                if modality == 'visual':
                    modality_features.append(torch.zeros(100, 417))
                elif modality == 'audio':
                    modality_features.append(torch.zeros(24))
                elif modality == 'text':
                    modality_features.append(torch.zeros(300))  # 默認unified維度

        # 堆疊成批次張量
        try:
            batched_features[modality] = torch.stack(modality_features, dim=0)
        except RuntimeError as e:
            print(f"Error stacking {modality} features: {e}")
            # 打印維度信息以便調試
            shapes = [f.shape for f in modality_features]
            print(f"Feature shapes: {shapes}")
            raise

    # 構建標籤張量
    batched_labels = torch.LongTensor(labels_list)

    return batched_features, batched_labels


def create_data_loaders(
    mapping_file: str = "features/trimodal_mapping.json",
    modalities: List[str] = ['visual', 'audio', 'text'],
    batch_size: int = 32,
    num_workers: int = 4,
    train_val_split: float = 0.8,
    text_embedding_type: str = 'unified',
    use_semantic_sampling: bool = False,
    visual_augment: bool = True,
    audio_dropout: float = 0.1,
    max_samples_per_word: Optional[int] = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建訓練、驗證、測試數據載入器

    Args:
        mapping_file: 三模態映射文件路徑
        modalities: 使用的模態列表
        batch_size: 批次大小
        num_workers: 數據載入進程數
        train_val_split: 訓練集比例 (剩餘為驗證集)
        text_embedding_type: 文字嵌入類型
        use_semantic_sampling: 是否使用語義平衡採樣
        visual_augment: 是否使用視覺數據增強
        audio_dropout: 音訊dropout比例
        max_samples_per_word: 每個詞彙最大樣本數
        pin_memory: 是否使用pin memory優化

    Returns:
        train_loader, val_loader, test_loader
    """

    # 創建數據集
    train_dataset = TriModalDataset(
        mapping_file=mapping_file,
        mode='train',
        modalities=modalities,
        visual_augment=visual_augment,
        audio_dropout=audio_dropout,
        text_embedding_type=text_embedding_type,
        semantic_sampling=use_semantic_sampling,
        max_samples_per_word=max_samples_per_word
    )

    val_dataset = TriModalDataset(
        mapping_file=mapping_file,
        mode='val',
        modalities=modalities,
        visual_augment=False,  # 驗證集不使用數據增強
        audio_dropout=0.0,
        text_embedding_type=text_embedding_type,
        semantic_sampling=use_semantic_sampling,
        max_samples_per_word=max_samples_per_word
    )

    test_dataset = TriModalDataset(
        mapping_file=mapping_file,
        mode='test',
        modalities=modalities,
        visual_augment=False,  # 測試集不使用數據增強
        audio_dropout=0.0,
        text_embedding_type=text_embedding_type,
        semantic_sampling=use_semantic_sampling,
        max_samples_per_word=max_samples_per_word
    )

    # 打印數據集統計資訊
    print("\n📊 數據集統計:")
    for dataset, name in [(train_dataset, 'Train'), (val_dataset, 'Val'), (test_dataset, 'Test')]:
        stats = dataset.get_statistics()
        print(f"{name:>5}: {stats['total_samples']:>5} 樣本, "
              f"平均每詞 {stats['avg_samples_per_word']:.1f} 樣本")

    # 創建數據載入器
    common_loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_multimodal_batch,
        'pin_memory': pin_memory and torch.cuda.is_available()
    }

    # 訓練集載入器
    if use_semantic_sampling and train_dataset.semantic_sampling:
        # 使用語義平衡採樣器
        semantic_sampler = SemanticBalancedSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=semantic_sampler,
            **{k: v for k, v in common_loader_args.items() if k != 'batch_size'}
        )
    else:
        # 使用隨機採樣
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_loader_args
        )

    # 驗證集和測試集載入器 (不使用特殊採樣)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args
    )

    return train_loader, val_loader, test_loader


def create_single_modal_loaders(
    modality: str,
    mapping_file: str = "features/trimodal_mapping.json",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建單模態數據載入器 (用於階段1訓練)

    Args:
        modality: 單一模態名稱 ('visual', 'audio', 'text')
        其他參數同 create_data_loaders

    Returns:
        train_loader, val_loader, test_loader (僅包含指定模態)
    """
    return create_data_loaders(
        mapping_file=mapping_file,
        modalities=[modality],
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )


def create_dual_modal_loaders(
    modality_pair: List[str],
    mapping_file: str = "features/trimodal_mapping.json",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建雙模態數據載入器 (用於階段2訓練)

    Args:
        modality_pair: 模態對，如 ['visual', 'audio']
        其他參數同 create_data_loaders

    Returns:
        train_loader, val_loader, test_loader (僅包含指定模態對)
    """
    if len(modality_pair) != 2:
        raise ValueError("modality_pair must contain exactly 2 modalities")

    return create_data_loaders(
        mapping_file=mapping_file,
        modalities=modality_pair,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )


class DataLoaderFactory:
    """
    數據載入器工廠類
    簡化不同階段的數據載入器創建
    """

    def __init__(
        self,
        mapping_file: str = "features/trimodal_mapping.json",
        batch_size: int = 32,
        num_workers: int = 4,
        text_embedding_type: str = 'unified',
        **default_kwargs
    ):
        self.mapping_file = mapping_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_embedding_type = text_embedding_type
        self.default_kwargs = default_kwargs

    def stage1_loaders(self, modality: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """階段1: 單模態訓練載入器"""
        return create_single_modal_loaders(
            modality=modality,
            mapping_file=self.mapping_file,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            text_embedding_type=self.text_embedding_type,
            **self.default_kwargs
        )

    def stage2_loaders(self, modality_pair: List[str]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """階段2: 雙模態訓練載入器"""
        return create_dual_modal_loaders(
            modality_pair=modality_pair,
            mapping_file=self.mapping_file,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            text_embedding_type=self.text_embedding_type,
            **self.default_kwargs
        )

    def stage3_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """階段3: 三模態訓練載入器"""
        return create_data_loaders(
            mapping_file=self.mapping_file,
            modalities=['visual', 'audio', 'text'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            text_embedding_type=self.text_embedding_type,
            **self.default_kwargs
        )


def test_data_loading():
    """
    測試數據載入功能
    用於驗證數據載入器是否正常工作
    """
    print("🧪 測試數據載入器...")

    try:
        # 測試三模態載入器
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=8,
            num_workers=0  # 測試時使用單進程
        )

        print("✅ 數據載入器創建成功")

        # 測試批次載入
        for i, (features, labels) in enumerate(train_loader):
            print(f"✅ 批次 {i+1}:")
            print(f"   標籤形狀: {labels.shape}")

            for modality, feature_tensor in features.items():
                print(f"   {modality:>6} 特徵: {feature_tensor.shape}")

            if i >= 2:  # 只測試前3個批次
                break

        print("✅ 數據載入測試完成!")

    except Exception as e:
        print(f"❌ 數據載入測試失敗: {e}")
        raise


if __name__ == "__main__":
    test_data_loading()