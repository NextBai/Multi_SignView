"""
Multi-Modal Sign Language Dataset
三模態手語數據集實作
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

class TriModalDataset(Dataset):
    """
    三模態手語數據集

    支援特性:
    - 靈活的模態組合: visual, audio, text
    - 動態數據增強: 時間抖動、噪音注入
    - 批次載入優化: 預載入常用特徵
    - 詞彙語義分類: 根據語義類別進行採樣
    """

    def __init__(
        self,
        mapping_file: str = "features/trimodal_mapping.json",
        mode: str = 'train',
        modalities: List[str] = ['visual', 'audio', 'text'],
        visual_augment: bool = True,
        audio_dropout: float = 0.1,
        text_embedding_type: str = 'unified',
        semantic_sampling: bool = False,
        max_samples_per_word: Optional[int] = None
    ):
        """
        初始化三模態數據集

        Args:
            mapping_file: 三模態映射文件路徑
            mode: 數據集模式 ('train', 'val', 'test')
            modalities: 使用的模態列表
            visual_augment: 是否使用視覺數據增強
            audio_dropout: 音訊dropout比例
            text_embedding_type: 文字嵌入類型 ('unified', 'word2vec', 'fasttext', 'bert')
            semantic_sampling: 是否使用語義平衡採樣
            max_samples_per_word: 每個詞彙最大樣本數 (用於數據平衡)
        """
        self.mapping_file = mapping_file
        self.mode = mode
        self.modalities = modalities
        self.visual_augment = visual_augment and (mode == 'train')
        self.audio_dropout = audio_dropout if mode == 'train' else 0.0
        self.text_embedding_type = text_embedding_type
        self.semantic_sampling = semantic_sampling
        self.max_samples_per_word = max_samples_per_word

        # 載入映射文件
        self._load_mapping()

        # 建立詞彙索引
        self._build_vocabulary()

        # 構建樣本列表
        self._build_sample_list()

        # 載入文字嵌入 (如果需要)
        if 'text' in self.modalities:
            self._load_text_embeddings()

        # 載入語義分類 (如果使用語義採樣)
        if self.semantic_sampling:
            self._load_semantic_categories()

        print(f"📊 數據集初始化完成:")
        print(f"   模式: {self.mode}")
        print(f"   模態: {', '.join(self.modalities)}")
        print(f"   樣本數: {len(self.samples)}")
        print(f"   詞彙數: {len(self.words)}")

    def _load_mapping(self):
        """載入三模態映射文件"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到映射文件: {self.mapping_file}")
        except json.JSONDecodeError:
            raise ValueError(f"映射文件格式錯誤: {self.mapping_file}")

    def _build_vocabulary(self):
        """建立詞彙表和索引"""
        self.words = list(self.mapping.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.num_classes = len(self.words)

    def _build_sample_list(self):
        """構建樣本列表 (word, sample_idx)"""
        self.samples = []

        for word in self.words:
            # 檢查視覺特徵目錄
            visual_info = self.mapping[word]['modalities']['visual']
            if not visual_info['available']:
                warnings.warn(f"詞彙 '{word}' 視覺特徵不可用，跳過")
                continue

            visual_dir = Path(visual_info['mediapipe_dir'])
            if not visual_dir.exists():
                warnings.warn(f"視覺特徵目錄不存在: {visual_dir}")
                continue

            # 獲取該詞彙的所有樣本文件
            visual_files = sorted(list(visual_dir.glob('*.npy')))

            # 限制每個詞彙的樣本數量 (數據平衡)
            if self.max_samples_per_word:
                visual_files = visual_files[:self.max_samples_per_word]

            # 根據模式分割數據
            if self.mode == 'train':
                # 訓練集: 前80%
                split_idx = int(0.8 * len(visual_files))
                selected_files = visual_files[:split_idx]
            elif self.mode == 'val':
                # 驗證集: 80%-90%
                split_start = int(0.8 * len(visual_files))
                split_end = int(0.9 * len(visual_files))
                selected_files = visual_files[split_start:split_end]
            else:  # test
                # 測試集: 後10%
                split_idx = int(0.9 * len(visual_files))
                selected_files = visual_files[split_idx:]

            # 添加樣本到列表
            for i in range(len(selected_files)):
                self.samples.append((word, i, len(selected_files)))

    def _load_text_embeddings(self):
        """載入文字嵌入矩陣"""
        embedding_files = {
            'unified': 'features/text_embeddings/unified_embeddings.npy',
            'word2vec': 'features/text_embeddings/word2vec_embeddings.npy',
            'fasttext': 'features/text_embeddings/fasttext_embeddings.npy',
            'bert': 'features/text_embeddings/bert_embeddings.npy'
        }

        if self.text_embedding_type not in embedding_files:
            raise ValueError(f"不支援的文字嵌入類型: {self.text_embedding_type}")

        try:
            self.text_embeddings = np.load(embedding_files[self.text_embedding_type])
            print(f"✅ 載入文字嵌入: {self.text_embedding_type} {self.text_embeddings.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文字嵌入文件: {embedding_files[self.text_embedding_type]}")

    def _load_semantic_categories(self):
        """載入語義分類資訊"""
        try:
            with open('features/semantic_features/semantic_analysis_fixed.json', 'r', encoding='utf-8') as f:
                semantic_data = json.load(f)
            self.semantic_categories = semantic_data.get('semantic_categories', {})
            print(f"✅ 載入語義分類: {len(self.semantic_categories)}個類別")
        except FileNotFoundError:
            warnings.warn("找不到語義分類文件，禁用語義採樣")
            self.semantic_sampling = False

    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        獲取單個樣本

        Returns:
            features: 字典，包含選定模態的特徵
            label: 詞彙標籤索引
        """
        word, sample_idx, total_samples = self.samples[idx]
        label = self.word_to_idx[word]

        features = {}

        # 載入視覺特徵
        if 'visual' in self.modalities:
            features['visual'] = self._load_visual_features(word, sample_idx, total_samples)

        # 載入音訊特徵
        if 'audio' in self.modalities:
            features['audio'] = self._load_audio_features(word)

        # 載入文字特徵
        if 'text' in self.modalities:
            features['text'] = self._load_text_features(label)

        return features, label

    def _load_visual_features(self, word: str, sample_idx: int, total_samples: int) -> torch.Tensor:
        """載入視覺特徵"""
        visual_dir = Path(self.mapping[word]['modalities']['visual']['mediapipe_dir'])
        visual_files = sorted(list(visual_dir.glob('*.npy')))

        # 根據數據分割選擇正確的文件
        if self.mode == 'train':
            split_idx = int(0.8 * len(visual_files))
            available_files = visual_files[:split_idx]
        elif self.mode == 'val':
            split_start = int(0.8 * len(visual_files))
            split_end = int(0.9 * len(visual_files))
            available_files = visual_files[split_start:split_end]
        else:  # test
            split_idx = int(0.9 * len(visual_files))
            available_files = visual_files[split_idx:]

        # 載入視覺數據
        visual_data = np.load(available_files[sample_idx])  # (100, 417)

        # 數據增強 (僅訓練模式)
        if self.visual_augment:
            visual_data = self._augment_visual(visual_data)

        return torch.FloatTensor(visual_data)

    def _load_audio_features(self, word: str) -> torch.Tensor:
        """載入音訊特徵"""
        # 使用normalized版本的音訊特徵
        audio_file = f"features/audio_features/{word}_normalized_24d.npy"

        try:
            audio_data = np.load(audio_file)  # (24,)
        except FileNotFoundError:
            warnings.warn(f"找不到音訊特徵文件: {audio_file}")
            # 返回零向量作為fallback
            audio_data = np.zeros(24, dtype=np.float32)

        # 音訊dropout (僅訓練模式)
        if self.audio_dropout > 0:
            dropout_mask = np.random.random(audio_data.shape) > self.audio_dropout
            audio_data = audio_data * dropout_mask

        return torch.FloatTensor(audio_data)

    def _load_text_features(self, label: int) -> torch.Tensor:
        """載入文字特徵"""
        text_data = self.text_embeddings[label]  # (300,) or (768,)
        return torch.FloatTensor(text_data)

    def _augment_visual(self, visual_data: np.ndarray) -> np.ndarray:
        """
        視覺數據增強

        增強策略:
        1. 時間軸隨機偏移 (±5幀)
        2. 高斯噪音注入 (σ=0.01)
        3. 隨機幀dropout (概率0.05)
        4. 空間座標微調 (±2像素)
        """
        augmented_data = visual_data.copy()

        # 1. 時間軸隨機偏移
        if random.random() < 0.5:
            shift = random.randint(-5, 5)
            if shift != 0:
                augmented_data = np.roll(augmented_data, shift, axis=0)

        # 2. 高斯噪音注入
        if random.random() < 0.3:
            noise_std = random.uniform(0.005, 0.015)
            noise = np.random.normal(0, noise_std, augmented_data.shape)
            augmented_data = augmented_data + noise

        # 3. 隨機幀dropout
        if random.random() < 0.2:
            dropout_prob = random.uniform(0.02, 0.08)
            dropout_mask = np.random.random(augmented_data.shape[0]) > dropout_prob
            for i, keep in enumerate(dropout_mask):
                if not keep and i > 0:
                    # 使用前一幀數據替代
                    augmented_data[i] = augmented_data[i-1]

        # 4. 空間座標微調 (僅對x,y座標，保持z座標和visibility不變)
        if random.random() < 0.4:
            # MediaPipe特徵結構: 每3個維度為一組 (x,y,z)
            coordinate_noise = np.random.normal(0, 0.002, augmented_data.shape)
            # 只對x,y座標添加噪音 (每組的前兩個維度)
            for i in range(0, augmented_data.shape[1], 3):
                if i+1 < augmented_data.shape[1]:
                    augmented_data[:, i:i+2] += coordinate_noise[:, i:i+2]

        return augmented_data

    def get_word_samples(self, word: str) -> List[Tuple[Dict[str, torch.Tensor], int]]:
        """獲取特定詞彙的所有樣本"""
        word_samples = []
        for i, (sample_word, _, _) in enumerate(self.samples):
            if sample_word == word:
                features, label = self[i]
                word_samples.append((features, label))
        return word_samples

    def get_semantic_category_samples(self, category: str) -> List[Tuple[Dict[str, torch.Tensor], int]]:
        """根據語義類別獲取樣本"""
        if not self.semantic_sampling:
            raise ValueError("語義採樣未啟用")

        if category not in self.semantic_categories:
            raise ValueError(f"未知語義類別: {category}")

        category_words = self.semantic_categories[category]
        category_samples = []

        for word in category_words:
            if word in self.word_to_idx:
                word_samples = self.get_word_samples(word)
                category_samples.extend(word_samples)

        return category_samples

    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """獲取數據集統計資訊"""
        stats = {
            'total_samples': len(self.samples),
            'num_classes': self.num_classes,
            'modalities': self.modalities,
            'mode': self.mode,
            'samples_per_word': {}
        }

        # 統計每個詞彙的樣本數
        for word in self.words:
            word_count = sum(1 for sample_word, _, _ in self.samples if sample_word == word)
            stats['samples_per_word'][word] = word_count

        # 計算統計量
        sample_counts = list(stats['samples_per_word'].values())
        stats['avg_samples_per_word'] = np.mean(sample_counts)
        stats['min_samples_per_word'] = np.min(sample_counts)
        stats['max_samples_per_word'] = np.max(sample_counts)
        stats['std_samples_per_word'] = np.std(sample_counts)

        return stats


class SemanticBalancedSampler:
    """
    語義平衡採樣器
    確保不同語義類別的樣本在訓練中均勻分布
    """

    def __init__(self, dataset: TriModalDataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size

        if not dataset.semantic_sampling:
            raise ValueError("數據集必須啟用語義採樣")

        self._build_category_indices()

    def _build_category_indices(self):
        """建立語義類別索引"""
        self.category_indices = {}

        for category, words in self.dataset.semantic_categories.items():
            indices = []
            for i, (word, _, _) in enumerate(self.dataset.samples):
                if word in words:
                    indices.append(i)
            self.category_indices[category] = indices

    def __iter__(self):
        """生成平衡的批次索引"""
        categories = list(self.category_indices.keys())
        category_iterators = {
            cat: iter(np.random.permutation(indices))
            for cat, indices in self.category_indices.items()
        }

        while True:
            batch_indices = []

            for _ in range(self.batch_size):
                # 隨機選擇語義類別
                category = random.choice(categories)

                try:
                    idx = next(category_iterators[category])
                    batch_indices.append(idx)
                except StopIteration:
                    # 重新洗牌該類別
                    category_iterators[category] = iter(
                        np.random.permutation(self.category_indices[category])
                    )
                    idx = next(category_iterators[category])
                    batch_indices.append(idx)

            yield batch_indices