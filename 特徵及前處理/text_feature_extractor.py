#!/usr/bin/env python3
"""
文字特徵提取器 - Multi_SignView 多模態手語辨識系統
整合 Word2Vec、FastText、BERT 多層次語義特徵提取
支援30個手語詞彙的語義分析和相似度計算
構建詞彙語義空間，輔助手語分類任務

技術特點：
- Word2Vec(300) + FastText(300) + BERT(768) 多層次詞嵌入
- 30×30 詞彙語義相似度矩陣計算
- 語義聚類分析 (動作類、物品類、人物類等)
- 同義詞擴展和語義網絡構建
- 支援中英文混合詞彙處理

Author: Claude Code + Multi_SignView Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# NLP 核心依賴
import gensim
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors

# Transformers 和 BERT
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# NLTK 自然語言處理
import nltk
try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("⚠️  NLTK 資源下載失敗，部分功能可能受限")

# 路徑配置
PROJECT_ROOT = Path(__file__).parent
OUTPUT_ROOT = PROJECT_ROOT / "features" / "text_embeddings"
SEMANTIC_OUTPUT_ROOT = PROJECT_ROOT / "features" / "semantic_features"
MODEL_DIR = PROJECT_ROOT / "model" / "text_models"

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 30個手語詞彙列表
SIGN_LANGUAGE_VOCABULARY = [
    'again', 'bird', 'book', 'computer', 'cousin', 'deaf', 'drink', 'eat',
    'finish', 'fish', 'friend', 'good', 'happy', 'learn', 'like', 'mother',
    'need', 'nice', 'no', 'orange', 'school', 'sister', 'student', 'table',
    'teacher', 'tired', 'want', 'what', 'white', 'yes'
]


class VocabularyManager:
    """
    詞彙管理器

    負責管理30個手語詞彙的語義分析、聚類、相似度計算
    """

    def __init__(self, vocabulary: List[str] = SIGN_LANGUAGE_VOCABULARY):
        """
        初始化詞彙管理器

        Args:
            vocabulary: 手語詞彙列表
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}

        # 語義分析結果
        self.semantic_clusters = {}
        self.similarity_matrix = None
        self.synonyms_dict = {}

        print(f"📝 VocabularyManager 初始化完成")
        print(f"   - 詞彙數量: {self.vocab_size}")
        print(f"   - 詞彙列表: {', '.join(vocabulary[:10])}..." if len(vocabulary) > 10 else f"   - 詞彙列表: {', '.join(vocabulary)}")

    def analyze_semantic_categories(self) -> Dict[str, List[str]]:
        """
        分析詞彙的語義分類

        Returns:
            語義分類字典
        """
        try:
            # 手動定義語義分類 (基於手語語義學)
            semantic_categories = {
                '動作類': ['again', 'drink', 'eat', 'finish', 'learn', 'like', 'need', 'want'],
                '人物類': ['cousin', 'deaf', 'friend', 'mother', 'sister', 'student', 'teacher'],
                '物品類': ['bird', 'book', 'computer', 'fish', 'orange', 'table'],
                '狀態類': ['good', 'happy', 'nice', 'tired', 'white'],
                '場所類': ['school'],
                '疑問類': ['what'],
                '肯定否定類': ['no', 'yes']
            }

            self.semantic_clusters = semantic_categories

            print(f"📊 語義分類完成:")
            for category, words in semantic_categories.items():
                print(f"   - {category}: {len(words)}個詞彙 -> {', '.join(words)}")

            return semantic_categories

        except Exception as e:
            print(f"❌ 語義分類失敗: {str(e)}")
            return {}

    def extract_synonyms(self) -> Dict[str, List[str]]:
        """
        使用 WordNet 提取同義詞

        Returns:
            同義詞字典
        """
        try:
            synonyms_dict = {}

            for word in self.vocabulary:
                synonyms = set()

                try:
                    # 使用 WordNet 查找同義詞
                    synsets = wn.synsets(word)
                    for synset in synsets:
                        for lemma in synset.lemmas():
                            synonym = lemma.name().replace('_', ' ').lower()
                            if synonym != word:
                                synonyms.add(synonym)

                    synonyms_dict[word] = list(synonyms)[:5]  # 限制前5個同義詞

                except:
                    synonyms_dict[word] = []

            self.synonyms_dict = synonyms_dict

            print(f"🔗 同義詞提取完成:")
            for word, syns in list(synonyms_dict.items())[:5]:
                if syns:
                    print(f"   - {word}: {', '.join(syns)}")

            return synonyms_dict

        except Exception as e:
            print(f"❌ 同義詞提取失敗: {str(e)}")
            return {}

    def compute_semantic_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        計算詞彙語義相似度矩陣

        Args:
            embeddings: 詞彙嵌入矩陣 (vocab_size, embedding_dim)

        Returns:
            相似度矩陣 (vocab_size, vocab_size)
        """
        try:
            # 計算餘弦相似度
            cosine_sim = cosine_similarity(embeddings)

            # 計算歐氏距離 (轉換為相似度)
            euclidean_dist = euclidean_distances(embeddings)
            euclidean_sim = 1 / (1 + euclidean_dist)  # 距離轉相似度

            # 加權融合兩種相似度
            similarity_matrix = 0.7 * cosine_sim + 0.3 * euclidean_sim

            self.similarity_matrix = similarity_matrix

            print(f"📊 相似度矩陣計算完成: {similarity_matrix.shape}")

            return similarity_matrix

        except Exception as e:
            print(f"❌ 相似度矩陣計算失敗: {str(e)}")
            return np.eye(self.vocab_size)

    def find_most_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找最相似的詞彙

        Args:
            word: 查詢詞彙
            top_k: 返回前K個相似詞

        Returns:
            相似詞彙列表 [(詞彙, 相似度), ...]
        """
        try:
            if word not in self.word_to_idx or self.similarity_matrix is None:
                return []

            word_idx = self.word_to_idx[word]
            similarities = self.similarity_matrix[word_idx]

            # 排除自己，取前K個
            sorted_indices = np.argsort(similarities)[::-1][1:top_k+1]

            similar_words = [
                (self.idx_to_word[idx], similarities[idx])
                for idx in sorted_indices
            ]

            return similar_words

        except Exception as e:
            print(f"❌ 相似詞查找失敗: {str(e)}")
            return []

    def visualize_semantic_space(self, embeddings: np.ndarray, save_path: Optional[Path] = None):
        """
        視覺化詞彙語義空間

        Args:
            embeddings: 詞彙嵌入矩陣
            save_path: 儲存路徑
        """
        try:
            # 使用 PCA 降維到 2D
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)

            # 創建語義分類顏色映射
            category_colors = {
                '動作類': 'red',
                '人物類': 'blue',
                '物品類': 'green',
                '狀態類': 'orange',
                '場所類': 'purple',
                '疑問類': 'brown',
                '肯定否定類': 'pink'
            }

            # 繪製散點圖
            plt.figure(figsize=(12, 8))

            for category, words in self.semantic_clusters.items():
                color = category_colors.get(category, 'gray')
                indices = [self.word_to_idx[word] for word in words if word in self.word_to_idx]

                if indices:
                    x_coords = embeddings_2d[indices, 0]
                    y_coords = embeddings_2d[indices, 1]
                    plt.scatter(x_coords, y_coords, c=color, label=category, alpha=0.7, s=100)

                    # 添加詞彙標籤
                    for i, word in enumerate(words):
                        if word in self.word_to_idx:
                            idx = self.word_to_idx[word]
                            plt.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                                       fontsize=8, ha='center', va='bottom')

            plt.title('手語詞彙語義空間分布 (PCA 2D)', fontsize=16)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path / 'semantic_space_visualization.png', dpi=300, bbox_inches='tight')
                print(f"📊 語義空間視覺化已儲存: {save_path / 'semantic_space_visualization.png'}")

            plt.show()

        except Exception as e:
            print(f"❌ 語義空間視覺化失敗: {str(e)}")


class TextFeatureExtractor:
    """
    多層次文字語義特徵提取器

    整合 Word2Vec、FastText、BERT 等多種詞嵌入技術
    """

    def __init__(self,
                 vocabulary: List[str] = SIGN_LANGUAGE_VOCABULARY,
                 bert_model_name: str = 'bert-base-uncased',
                 embedding_dims: Dict[str, int] = None):
        """
        初始化文字特徵提取器

        Args:
            vocabulary: 詞彙列表
            bert_model_name: BERT 模型名稱
            embedding_dims: 各種嵌入的維度配置
        """
        self.vocabulary = vocabulary
        self.vocab_manager = VocabularyManager(vocabulary)
        self.bert_model_name = bert_model_name

        # 預設嵌入維度
        if embedding_dims is None:
            embedding_dims = {
                'word2vec': 300,
                'fasttext': 300,
                'bert': 768
            }
        self.embedding_dims = embedding_dims

        # 模型儲存路徑
        self.model_dir = MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 載入的模型
        self.word2vec_model = None
        self.fasttext_model = None
        self.bert_tokenizer = None
        self.bert_model = None

        # 詞彙嵌入矩陣
        self.embeddings = {}

        print(f"📝 TextFeatureExtractor 初始化完成")
        print(f"   - 詞彙數量: {len(vocabulary)}")
        print(f"   - BERT 模型: {bert_model_name}")
        print(f"   - 嵌入維度: {embedding_dims}")

    def load_pretrained_embeddings(self):
        """
        載入預訓練的詞嵌入模型
        """
        try:
            print("🔄 正在載入預訓練詞嵌入模型...")

            # 1. 載入 Word2Vec (使用 Google News 預訓練模型)
            try:
                word2vec_path = self.model_dir / "GoogleNews-vectors-negative300.bin"
                if word2vec_path.exists():
                    print("   📚 載入 Word2Vec 模型...")
                    self.word2vec_model = KeyedVectors.load_word2vec_format(
                        str(word2vec_path), binary=True, limit=500000
                    )
                else:
                    print("   ⚠️  Word2Vec 預訓練模型未找到，將使用隨機初始化")
                    self.word2vec_model = None
            except Exception as e:
                print(f"   ❌ Word2Vec 載入失敗: {str(e)}")
                self.word2vec_model = None

            # 2. 載入 FastText (使用 Common Crawl 預訓練模型)
            try:
                fasttext_path = self.model_dir / "cc.en.300.vec"
                if fasttext_path.exists():
                    print("   📚 載入 FastText 模型...")
                    self.fasttext_model = KeyedVectors.load_word2vec_format(
                        str(fasttext_path), binary=False, limit=500000
                    )
                else:
                    print("   ⚠️  FastText 預訓練模型未找到，將使用隨機初始化")
                    self.fasttext_model = None
            except Exception as e:
                print(f"   ❌ FastText 載入失敗: {str(e)}")
                self.fasttext_model = None

            # 3. 載入 BERT
            try:
                print(f"   📚 載入 BERT 模型: {self.bert_model_name}")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                self.bert_model.eval()

                # 檢查是否有 GPU
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.bert_model.to(self.device)
                print(f"   🖥️  BERT 運行設備: {self.device}")

            except Exception as e:
                print(f"   ❌ BERT 載入失敗: {str(e)}")
                self.bert_tokenizer = None
                self.bert_model = None

            print("✅ 預訓練模型載入完成")

        except Exception as e:
            print(f"❌ 預訓練模型載入失敗: {str(e)}")

    def extract_word2vec_embeddings(self) -> np.ndarray:
        """
        提取 Word2Vec 詞嵌入

        Returns:
            Word2Vec 嵌入矩陣 (vocab_size, 300)
        """
        try:
            embeddings = []

            for word in self.vocabulary:
                if self.word2vec_model and word in self.word2vec_model:
                    embedding = self.word2vec_model[word]
                else:
                    # 隨機初始化未找到的詞彙
                    embedding = np.random.normal(0, 0.1, self.embedding_dims['word2vec'])

                embeddings.append(embedding)

            embeddings_matrix = np.array(embeddings, dtype=np.float32)

            print(f"📊 Word2Vec 嵌入提取完成: {embeddings_matrix.shape}")
            return embeddings_matrix

        except Exception as e:
            print(f"❌ Word2Vec 嵌入提取失敗: {str(e)}")
            return np.random.normal(0, 0.1, (len(self.vocabulary), self.embedding_dims['word2vec']))

    def extract_fasttext_embeddings(self) -> np.ndarray:
        """
        提取 FastText 詞嵌入

        Returns:
            FastText 嵌入矩陣 (vocab_size, 300)
        """
        try:
            embeddings = []

            for word in self.vocabulary:
                if self.fasttext_model and word in self.fasttext_model:
                    embedding = self.fasttext_model[word]
                else:
                    # 隨機初始化未找到的詞彙
                    embedding = np.random.normal(0, 0.1, self.embedding_dims['fasttext'])

                embeddings.append(embedding)

            embeddings_matrix = np.array(embeddings, dtype=np.float32)

            print(f"📊 FastText 嵌入提取完成: {embeddings_matrix.shape}")
            return embeddings_matrix

        except Exception as e:
            print(f"❌ FastText 嵌入提取失敗: {str(e)}")
            return np.random.normal(0, 0.1, (len(self.vocabulary), self.embedding_dims['fasttext']))

    def extract_bert_embeddings(self) -> np.ndarray:
        """
        提取 BERT 詞嵌入

        Returns:
            BERT 嵌入矩陣 (vocab_size, 768)
        """
        try:
            if self.bert_model is None or self.bert_tokenizer is None:
                print("⚠️  BERT 模型未載入，使用隨機初始化")
                return np.random.normal(0, 0.1, (len(self.vocabulary), self.embedding_dims['bert']))

            embeddings = []

            with torch.no_grad():
                for word in self.vocabulary:
                    # Tokenize 詞彙
                    inputs = self.bert_tokenizer(
                        word,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    # 獲取 BERT 輸出
                    outputs = self.bert_model(**inputs)

                    # 使用 [CLS] token 的表示作為詞彙嵌入
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embedding.squeeze())

            embeddings_matrix = np.array(embeddings, dtype=np.float32)

            print(f"📊 BERT 嵌入提取完成: {embeddings_matrix.shape}")
            return embeddings_matrix

        except Exception as e:
            print(f"❌ BERT 嵌入提取失敗: {str(e)}")
            return np.random.normal(0, 0.1, (len(self.vocabulary), self.embedding_dims['bert']))

    def extract_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        提取所有類型的詞嵌入

        Returns:
            所有嵌入字典
        """
        print("🔄 開始提取多層次詞嵌入...")

        # 載入預訓練模型
        self.load_pretrained_embeddings()

        # 提取各種嵌入
        self.embeddings = {
            'word2vec': self.extract_word2vec_embeddings(),
            'fasttext': self.extract_fasttext_embeddings(),
            'bert': self.extract_bert_embeddings()
        }

        print("✅ 多層次詞嵌入提取完成")
        return self.embeddings

    def create_unified_embedding(self, weights: Dict[str, float] = None) -> np.ndarray:
        """
        創建統一的多模態詞嵌入

        Args:
            weights: 各種嵌入的權重

        Returns:
            統一的詞嵌入矩陣
        """
        try:
            if weights is None:
                weights = {'word2vec': 0.3, 'fasttext': 0.3, 'bert': 0.4}

            # 標準化各種嵌入
            normalized_embeddings = {}
            for emb_type, embeddings in self.embeddings.items():
                scaler = StandardScaler()
                normalized_embeddings[emb_type] = scaler.fit_transform(embeddings)

            # 加權融合
            unified_embedding = None
            for emb_type, weight in weights.items():
                if emb_type in normalized_embeddings:
                    if unified_embedding is None:
                        unified_embedding = weight * normalized_embeddings[emb_type]
                    else:
                        # 維度對齊 (如果需要)
                        if unified_embedding.shape[1] != normalized_embeddings[emb_type].shape[1]:
                            # 使用 PCA 統一維度到 300
                            pca = PCA(n_components=300)
                            normalized_embeddings[emb_type] = pca.fit_transform(normalized_embeddings[emb_type])
                            if unified_embedding.shape[1] != 300:
                                pca_unified = PCA(n_components=300)
                                unified_embedding = pca_unified.fit_transform(unified_embedding)

                        unified_embedding += weight * normalized_embeddings[emb_type]

            print(f"📊 統一詞嵌入創建完成: {unified_embedding.shape}")
            print(f"   - 融合權重: {weights}")

            return unified_embedding

        except Exception as e:
            print(f"❌ 統一詞嵌入創建失敗: {str(e)}")
            return np.random.normal(0, 0.1, (len(self.vocabulary), 300))

    def analyze_semantic_structure(self) -> Dict:
        """
        分析詞彙語義結構

        Returns:
            語義分析結果
        """
        try:
            print("🔍 開始語義結構分析...")

            # 獲取統一嵌入
            unified_embedding = self.create_unified_embedding()

            # 語義分類
            semantic_categories = self.vocab_manager.analyze_semantic_categories()

            # 計算相似度矩陣
            similarity_matrix = self.vocab_manager.compute_semantic_similarity_matrix(unified_embedding)

            # 提取同義詞
            synonyms_dict = self.vocab_manager.extract_synonyms()

            # K-means 聚類
            n_clusters = len(semantic_categories)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(unified_embedding)

            # 建立聚類結果
            clustering_result = {}
            for i, word in enumerate(self.vocabulary):
                cluster_id = cluster_labels[i]
                if cluster_id not in clustering_result:
                    clustering_result[cluster_id] = []
                clustering_result[cluster_id].append(word)

            analysis_result = {
                'semantic_categories': semantic_categories,
                'similarity_matrix': similarity_matrix.tolist(),
                'synonyms_dict': synonyms_dict,
                'clustering_result': clustering_result,
                'unified_embedding': unified_embedding.tolist(),
                'vocabulary': self.vocabulary,
                'embedding_info': {
                    'dimensions': unified_embedding.shape,
                    'embedding_types': list(self.embeddings.keys()),
                    'total_words': len(self.vocabulary)
                }
            }

            print("✅ 語義結構分析完成")
            return analysis_result

        except Exception as e:
            print(f"❌ 語義結構分析失敗: {str(e)}")
            return {}

    def get_word_embedding(self, word: str, embedding_type: str = 'unified') -> Optional[np.ndarray]:
        """
        獲取特定詞彙的嵌入向量

        Args:
            word: 詞彙
            embedding_type: 嵌入類型 ('word2vec', 'fasttext', 'bert', 'unified')

        Returns:
            詞嵌入向量或 None
        """
        try:
            if word not in self.vocab_manager.word_to_idx:
                return None

            word_idx = self.vocab_manager.word_to_idx[word]

            if embedding_type == 'unified':
                unified_embedding = self.create_unified_embedding()
                return unified_embedding[word_idx]
            elif embedding_type in self.embeddings:
                return self.embeddings[embedding_type][word_idx]
            else:
                print(f"❌ 未知的嵌入類型: {embedding_type}")
                return None

        except Exception as e:
            print(f"❌ 詞嵌入獲取失敗: {str(e)}")
            return None


def run_text_feature_extraction(
    output_root: Optional[str] = None,
    semantic_output_root: Optional[str] = None,
    save_models: bool = True
) -> Dict:
    """
    執行完整的文字特徵提取流程

    Args:
        output_root: 嵌入輸出路徑
        semantic_output_root: 語義分析輸出路徑
        save_models: 是否儲存模型

    Returns:
        處理結果
    """
    # 路徑配置
    output_path = Path(output_root) if output_root else OUTPUT_ROOT
    semantic_path = Path(semantic_output_root) if semantic_output_root else SEMANTIC_OUTPUT_ROOT

    # 建立輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)
    semantic_path.mkdir(parents=True, exist_ok=True)

    print(f"📝 開始文字特徵提取")
    print(f"   📂 嵌入輸出路徑: {output_path}")
    print(f"   📂 語義分析輸出路徑: {semantic_path}")

    # 初始化特徵提取器
    extractor = TextFeatureExtractor()

    try:
        # 提取所有嵌入
        all_embeddings = extractor.extract_all_embeddings()

        # 儲存各種嵌入
        for emb_type, embeddings in all_embeddings.items():
            emb_file = output_path / f"{emb_type}_embeddings.npy"
            np.save(emb_file, embeddings)
            print(f"💾 {emb_type} 嵌入已儲存: {emb_file}")

        # 創建統一嵌入
        unified_embedding = extractor.create_unified_embedding()
        unified_file = output_path / "unified_embeddings.npy"
        np.save(unified_file, unified_embedding)
        print(f"💾 統一嵌入已儲存: {unified_file}")

        # 語義結構分析
        semantic_analysis = extractor.analyze_semantic_structure()

        # 儲存語義分析結果
        semantic_file = semantic_path / "semantic_analysis.json"
        with open(semantic_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_analysis, f, indent=2, ensure_ascii=False)
        print(f"💾 語義分析結果已儲存: {semantic_file}")

        # 儲存詞彙映射
        vocab_mapping = {
            'vocabulary': extractor.vocabulary,
            'word_to_idx': extractor.vocab_manager.word_to_idx,
            'idx_to_word': extractor.vocab_manager.idx_to_word
        }
        vocab_file = output_path / "vocabulary_mapping.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_mapping, f, indent=2, ensure_ascii=False)

        # 視覺化語義空間
        extractor.vocab_manager.visualize_semantic_space(unified_embedding, semantic_path)

        # 處理結果
        results = {
            'success': True,
            'vocabulary_size': len(extractor.vocabulary),
            'embedding_types': list(all_embeddings.keys()),
            'embedding_dimensions': {k: v.shape for k, v in all_embeddings.items()},
            'semantic_categories': len(semantic_analysis.get('semantic_categories', {})),
            'output_files': {
                'embeddings': str(output_path),
                'semantic_analysis': str(semantic_path)
            }
        }

        print(f"📝 文字特徵提取完成!")
        print(f"   ✅ 詞彙數量: {results['vocabulary_size']}")
        print(f"   ✅ 嵌入類型: {', '.join(results['embedding_types'])}")
        print(f"   ✅ 語義分類: {results['semantic_categories']}個類別")

        return results

    except Exception as e:
        print(f"❌ 文字特徵提取失敗: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # 執行文字特徵提取
    results = run_text_feature_extraction()