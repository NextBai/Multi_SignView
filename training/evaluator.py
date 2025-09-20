"""
Model Evaluation and Analysis
模型評估和分析工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

from models import MultiModalSignClassifier


class ModelEvaluator:
    """
    模型評估器
    支援多種評估指標和可視化
    """

    def __init__(
        self,
        model: MultiModalSignClassifier,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]
        self.model.eval()

    def evaluate(
        self,
        test_loader,
        return_predictions: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        完整模型評估

        Returns:
            evaluation_results: 評估結果字典
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_embeddings = {'visual': [], 'audio': [], 'text': []}
        all_logits = []

        with torch.no_grad():
            for features, labels in test_loader:
                # 移動數據到設備
                for k in features:
                    features[k] = features[k].to(self.device)
                labels = labels.to(self.device)

                # 前向傳播
                if return_embeddings:
                    logits, embeddings = self.model(features, return_embeddings=True)
                    for modality, emb in embeddings.items():
                        all_embeddings[modality].append(emb.cpu().numpy())
                else:
                    logits = self.model(features)

                # 記錄預測和標籤
                predictions = logits.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        # 轉換為numpy數組
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        # 計算評估指標
        results = self._compute_metrics(all_labels, all_predictions, all_logits)

        # 添加可選結果
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
            results['logits'] = all_logits

        if return_embeddings:
            # 合併嵌入
            for modality in all_embeddings:
                if all_embeddings[modality]:
                    all_embeddings[modality] = np.concatenate(all_embeddings[modality], axis=0)
            results['embeddings'] = all_embeddings

        return results

    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, Any]:
        """計算評估指標"""
        # 基礎指標
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )

        # Top-k準確率
        top3_accuracy = self._compute_topk_accuracy(labels, logits, k=3)
        top5_accuracy = self._compute_topk_accuracy(labels, logits, k=5)

        # 混淆矩陣
        conf_matrix = confusion_matrix(labels, predictions)

        # 每類別指標
        class_metrics = []
        for i in range(len(self.class_names)):
            class_metrics.append({
                'class_name': self.class_names[i],
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            })

        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'confusion_matrix': conf_matrix,
            'class_metrics': class_metrics,
            'classification_report': classification_report(
                labels, predictions,
                target_names=self.class_names,
                zero_division=0,
                output_dict=True
            )
        }

    def _compute_topk_accuracy(self, labels: np.ndarray, logits: np.ndarray, k: int) -> float:
        """計算Top-k準確率"""
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i]:
                correct += 1
        return correct / len(labels)

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """繪製混淆矩陣"""
        plt.figure(figsize=figsize)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_class_performance(
        self,
        class_metrics: List[Dict],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ):
        """繪製每類別性能"""
        class_names = [m['class_name'] for m in class_metrics]
        precision = [m['precision'] for m in class_metrics]
        recall = [m['recall'] for m in class_metrics]
        f1 = [m['f1'] for m in class_metrics]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_errors(
        self,
        test_loader,
        num_examples: int = 10
    ) -> Dict[str, List]:
        """錯誤案例分析"""
        self.model.eval()
        error_examples = []

        with torch.no_grad():
            for features, labels in test_loader:
                # 移動數據到設備
                for k in features:
                    features[k] = features[k].to(self.device)
                labels = labels.to(self.device)

                # 前向傳播
                logits = self.model(features)
                predictions = logits.argmax(dim=1)
                confidences = F.softmax(logits, dim=1)

                # 找出錯誤預測
                errors = predictions != labels
                error_indices = torch.where(errors)[0]

                for idx in error_indices:
                    if len(error_examples) >= num_examples:
                        break

                    error_examples.append({
                        'true_label': self.class_names[labels[idx].item()],
                        'predicted_label': self.class_names[predictions[idx].item()],
                        'confidence': confidences[idx][predictions[idx]].item(),
                        'true_confidence': confidences[idx][labels[idx]].item(),
                        'logits': logits[idx].cpu().numpy().tolist()
                    })

                if len(error_examples) >= num_examples:
                    break

        return {'error_examples': error_examples}


class CrossModalAnalyzer:
    """
    跨模態分析器
    分析不同模態組合的性能
    """

    def __init__(
        self,
        model: MultiModalSignClassifier,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]

    def modal_ablation_study(self, test_loader) -> Dict[str, Dict]:
        """模態消融研究"""
        modal_combinations = [
            ['visual'],
            ['audio'],
            ['text'],
            ['visual', 'audio'],
            ['visual', 'text'],
            ['audio', 'text'],
            ['visual', 'audio', 'text']
        ]

        results = {}

        for modalities in modal_combinations:
            print(f"評估模態組合: {'+'.join(modalities)}")

            # 設置模型使用的模態
            self.model.set_modalities(modalities)

            # 評估
            evaluator = ModelEvaluator(self.model, self.device, self.class_names)
            modal_results = evaluator.evaluate(test_loader)

            combination_name = '+'.join(modalities)
            results[combination_name] = {
                'accuracy': modal_results['accuracy'],
                'macro_f1': modal_results['macro_f1'],
                'top3_accuracy': modal_results['top3_accuracy']
            }

        return results

    def attention_weight_analysis(
        self,
        test_loader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """注意力權重分析"""
        self.model.set_modalities(['visual', 'audio', 'text'])
        self.model.eval()

        attention_weights = []
        predictions = []
        labels = []

        with torch.no_grad():
            sample_count = 0
            for features, label_batch in test_loader:
                if sample_count >= num_samples:
                    break

                # 移動數據到設備
                for k in features:
                    features[k] = features[k].to(self.device)
                label_batch = label_batch.to(self.device)

                # 獲取注意力權重
                logits, embeddings, attn_weights = self.model(
                    features,
                    return_embeddings=True,
                    return_attention_weights=True
                )

                predictions.extend(logits.argmax(dim=1).cpu().numpy())
                labels.extend(label_batch.cpu().numpy())
                attention_weights.append(attn_weights)

                sample_count += len(label_batch)

        # 分析注意力模式
        return self._analyze_attention_patterns(attention_weights, predictions, labels)

    def _analyze_attention_patterns(
        self,
        attention_weights: List[Dict],
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, Any]:
        """分析注意力模式"""
        # 統計各模態對的注意力權重
        modal_pair_weights = {}

        for attn_dict in attention_weights:
            for pair_name, weights in attn_dict.items():
                if pair_name not in modal_pair_weights:
                    modal_pair_weights[pair_name] = []
                modal_pair_weights[pair_name].append(weights.mean().item())

        # 計算平均注意力權重
        avg_weights = {}
        for pair_name, weights in modal_pair_weights.items():
            avg_weights[pair_name] = np.mean(weights)

        return {
            'average_attention_weights': avg_weights,
            'attention_weight_distribution': modal_pair_weights
        }

    def modality_importance_analysis(
        self,
        test_loader,
        semantic_categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """模態重要性分析"""
        if semantic_categories is None:
            # 默認語義分類
            semantic_categories = {
                '動作類': ['again', 'drink', 'eat', 'finish', 'learn', 'like', 'need', 'want'],
                '人物類': ['cousin', 'deaf', 'friend', 'mother', 'sister', 'student', 'teacher'],
                '物品類': ['bird', 'book', 'computer', 'fish', 'orange', 'table'],
                '狀態類': ['good', 'happy', 'tired', 'nice', 'no', 'yes', 'white', 'what', 'school']
            }

        # 為每個語義類別分析模態重要性
        category_analysis = {}

        for category, words in semantic_categories.items():
            category_results = self._analyze_category_modality_importance(
                test_loader, words, category
            )
            category_analysis[category] = category_results

        return category_analysis

    def _analyze_category_modality_importance(
        self,
        test_loader,
        category_words: List[str],
        category_name: str
    ) -> Dict[str, float]:
        """分析特定類別的模態重要性"""
        # 獲取類別對應的標籤索引
        category_indices = []
        for word in category_words:
            if word in self.class_names:
                category_indices.append(self.class_names.index(word))

        if not category_indices:
            return {}

        # 測試不同模態組合在該類別上的性能
        modal_combinations = [
            ['visual'],
            ['audio'],
            ['text'],
            ['visual', 'audio', 'text']
        ]

        category_performance = {}

        for modalities in modal_combinations:
            self.model.set_modalities(modalities)
            accuracy = self._evaluate_category_accuracy(test_loader, category_indices)
            combination_name = '+'.join(modalities)
            category_performance[combination_name] = accuracy

        return category_performance

    def _evaluate_category_accuracy(
        self,
        test_loader,
        category_indices: List[int]
    ) -> float:
        """評估特定類別的準確率"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                # 移動數據到設備
                for k in features:
                    features[k] = features[k].to(self.device)
                labels = labels.to(self.device)

                # 只考慮目標類別的樣本
                category_mask = torch.isin(labels, torch.tensor(category_indices).to(self.device))
                if not category_mask.any():
                    continue

                category_features = {k: v[category_mask] for k, v in features.items()}
                category_labels = labels[category_mask]

                # 前向傳播
                logits = self.model(category_features)
                predictions = logits.argmax(dim=1)

                correct += (predictions == category_labels).sum().item()
                total += len(category_labels)

        return correct / total if total > 0 else 0.0


def comprehensive_evaluation(
    model: MultiModalSignClassifier,
    test_loader,
    device: torch.device,
    class_names: List[str],
    save_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    全面評估模型性能

    Args:
        model: 多模態分類器
        test_loader: 測試數據載入器
        device: 計算設備
        class_names: 類別名稱列表
        save_dir: 結果保存目錄

    Returns:
        complete_results: 完整評估結果
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("🔍 開始全面模型評估...")

    # 1. 基礎性能評估
    print("📊 1. 基礎性能評估")
    evaluator = ModelEvaluator(model, device, class_names)
    basic_results = evaluator.evaluate(test_loader, return_predictions=True)

    # 保存混淆矩陣圖
    evaluator.plot_confusion_matrix(
        basic_results['confusion_matrix'],
        save_path=save_path / 'confusion_matrix.png'
    )

    # 保存每類別性能圖
    evaluator.plot_class_performance(
        basic_results['class_metrics'],
        save_path=save_path / 'class_performance.png'
    )

    # 2. 跨模態分析
    print("🔀 2. 跨模態分析")
    cross_modal_analyzer = CrossModalAnalyzer(model, device, class_names)
    ablation_results = cross_modal_analyzer.modal_ablation_study(test_loader)

    # 3. 注意力權重分析
    print("👁️ 3. 注意力權重分析")
    attention_results = cross_modal_analyzer.attention_weight_analysis(test_loader)

    # 4. 模態重要性分析
    print("🎯 4. 模態重要性分析")
    importance_results = cross_modal_analyzer.modality_importance_analysis(test_loader)

    # 5. 錯誤案例分析
    print("❌ 5. 錯誤案例分析")
    error_results = evaluator.analyze_errors(test_loader)

    # 整合所有結果
    complete_results = {
        'basic_performance': basic_results,
        'modal_ablation': ablation_results,
        'attention_analysis': attention_results,
        'modality_importance': importance_results,
        'error_analysis': error_results,
        'evaluation_summary': {
            'overall_accuracy': basic_results['accuracy'],
            'macro_f1': basic_results['macro_f1'],
            'top3_accuracy': basic_results['top3_accuracy'],
            'best_single_modal': max(
                [(k, v['accuracy']) for k, v in ablation_results.items()
                 if '+' not in k], key=lambda x: x[1]
            ),
            'best_dual_modal': max(
                [(k, v['accuracy']) for k, v in ablation_results.items()
                 if k.count('+') == 1], key=lambda x: x[1]
            )
        }
    }

    # 保存結果
    results_file = save_path / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 轉換numpy數組為列表以便JSON序列化
        json_results = complete_results.copy()
        if 'confusion_matrix' in json_results['basic_performance']:
            json_results['basic_performance']['confusion_matrix'] = \
                json_results['basic_performance']['confusion_matrix'].tolist()
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"📁 評估結果保存至: {save_path}")
    print(f"📊 整體準確率: {basic_results['accuracy']:.3f}")
    print(f"🏆 最佳三模態F1: {basic_results['macro_f1']:.3f}")

    return complete_results


if __name__ == "__main__":
    # 測試評估功能
    print("🧪 測試評估模組...")
    print("💡 要進行完整評估，使用 comprehensive_evaluation() 函數")