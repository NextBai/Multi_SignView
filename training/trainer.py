"""
Progressive Multi-Modal Training Framework
三階段漸進式多模態訓練框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import time
import os
from pathlib import Path
import json

from models import MultiModalSignClassifier, create_multimodal_classifier
from .utils import EarlyStopping, ModelCheckpoint, TrainingLogger


class BaseTrainer:
    """基礎訓練器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 優化器和學習率調度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=learning_rate * 0.01
        )

        # 損失函數
        self.criterion = nn.CrossEntropyLoss()

        # 訓練工具
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        self.checkpoint = ModelCheckpoint(save_dir=save_dir)
        self.logger = TrainingLogger()

    def train_epoch(self) -> Tuple[float, float]:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 移動數據到設備
            for k in features:
                features[k] = features[k].to(self.device)
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """驗證"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                # 移動數據到設備
                for k in features:
                    features[k] = features[k].to(self.device)
                labels = labels.to(self.device)

                # 前向傳播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # 統計
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, epochs: int, verbose: bool = True) -> Dict[str, List[float]]:
        """完整訓練流程"""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            start_time = time.time()

            # 訓練
            train_loss, train_acc = self.train_epoch()

            # 驗證
            val_loss, val_acc = self.validate()

            # 學習率調度
            self.scheduler.step()

            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 日誌記錄
            epoch_time = time.time() - start_time
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, "
                      f"Time={epoch_time:.1f}s")

            # 早停檢查
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

            # 模型檢查點
            is_best = self.checkpoint.update(val_acc, self.model, epoch)
            if is_best and verbose:
                print(f"🏆 New best model saved with val_acc={val_acc:.3f}")

        return history


class SingleModalTrainer(BaseTrainer):
    """
    單模態訓練器
    用於Stage 1: 單模態預訓練
    """

    def __init__(self, modality: str, **kwargs):
        self.modality = modality

        # 創建單模態模型
        model = create_multimodal_classifier(
            modalities=[modality],
            **kwargs.pop('model_config', {})
        )

        super().__init__(model=model, **kwargs)

    def train_epoch(self) -> Tuple[float, float]:
        """單模態訓練一個epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 移動數據到設備，只使用指定模態
            modal_features = {
                self.modality: features[self.modality].to(self.device)
            }
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(modal_features)
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy


class DualModalTrainer(BaseTrainer):
    """
    雙模態訓練器
    用於Stage 2: 雙模態對齊學習
    """

    def __init__(
        self,
        modality_pair: List[str],
        contrastive_weight: float = 0.1,
        **kwargs
    ):
        self.modality_pair = modality_pair
        self.contrastive_weight = contrastive_weight

        # 創建雙模態模型
        model = create_multimodal_classifier(
            modalities=modality_pair,
            use_contrastive_loss=True,
            **kwargs.pop('model_config', {})
        )

        super().__init__(model=model, **kwargs)

    def train_epoch(self) -> Tuple[float, float]:
        """雙模態訓練一個epoch（包含對比學習）"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 移動數據到設備，只使用指定模態對
            modal_features = {}
            for modality in self.modality_pair:
                modal_features[modality] = features[modality].to(self.device)
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs, embeddings = self.model(
                modal_features,
                return_embeddings=True
            )

            # 分類損失
            cls_loss = self.criterion(outputs, labels)

            # 對比損失
            contrastive_losses = self.model.compute_contrastive_loss(embeddings, labels)
            contrastive_loss = sum(contrastive_losses.values())

            # 總損失
            total_loss_batch = cls_loss + self.contrastive_weight * contrastive_loss

            # 反向傳播
            total_loss_batch.backward()
            self.optimizer.step()

            # 統計
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader)
        accuracy = correct / total

        print(f"  分類損失: {avg_cls_loss:.4f}, 對比損失: {avg_contrastive_loss:.4f}")
        return avg_loss, accuracy


class TriModalTrainer(BaseTrainer):
    """
    三模態訓練器
    用於Stage 3: 三模態融合訓練
    """

    def __init__(
        self,
        contrastive_weight: float = 0.1,
        consistency_weight: float = 0.05,
        **kwargs
    ):
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight

        # 創建三模態模型
        model = create_multimodal_classifier(
            modalities=['visual', 'audio', 'text'],
            use_contrastive_loss=True,
            **kwargs.pop('model_config', {})
        )

        super().__init__(model=model, **kwargs)

    def train_epoch(self) -> Tuple[float, float]:
        """三模態訓練一個epoch（包含對比學習和一致性正則化）"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        total_consistency_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 移動數據到設備
            for k in features:
                features[k] = features[k].to(self.device)
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs, embeddings, attention_weights = self.model(
                features,
                return_embeddings=True,
                return_attention_weights=True
            )

            # 分類損失
            cls_loss = self.criterion(outputs, labels)

            # 對比損失
            contrastive_losses = self.model.compute_contrastive_loss(embeddings, labels)
            contrastive_loss = sum(contrastive_losses.values())

            # 一致性損失（模態間預測一致性）
            consistency_loss = self._compute_consistency_loss(embeddings, labels)

            # 總損失
            total_loss_batch = (
                cls_loss +
                self.contrastive_weight * contrastive_loss +
                self.consistency_weight * consistency_loss
            )

            # 反向傳播
            total_loss_batch.backward()
            self.optimizer.step()

            # 統計
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_consistency_loss += consistency_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader)
        avg_consistency_loss = total_consistency_loss / len(self.train_loader)
        accuracy = correct / total

        print(f"  分類: {avg_cls_loss:.4f}, 對比: {avg_contrastive_loss:.4f}, 一致性: {avg_consistency_loss:.4f}")
        return avg_loss, accuracy

    def _compute_consistency_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """計算模態間一致性損失"""
        # 創建臨時分類器來測試各模態的預測一致性
        temp_classifier = nn.Linear(512, self.model.num_classes).to(self.device)

        consistency_losses = []
        modal_predictions = {}

        # 為每個模態計算預測
        for modality, emb in embeddings.items():
            modal_pred = temp_classifier(emb)
            modal_predictions[modality] = F.softmax(modal_pred, dim=-1)

        # 計算模態間預測的KL散度
        modalities = list(modal_predictions.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                pred_i = modal_predictions[modalities[i]]
                pred_j = modal_predictions[modalities[j]]

                # KL散度（雙向）
                kl_loss = F.kl_div(
                    pred_i.log(), pred_j, reduction='batchmean'
                ) + F.kl_div(
                    pred_j.log(), pred_i, reduction='batchmean'
                )
                consistency_losses.append(kl_loss)

        return torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0)


class ProgressiveTrainer:
    """
    漸進式三階段訓練協調器
    協調整個訓練流程
    """

    def __init__(
        self,
        data_loaders: Dict[str, DataLoader],
        device: torch.device,
        save_dir: str = "checkpoints",
        config: Optional[Dict] = None
    ):
        self.data_loaders = data_loaders
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        # 訓練歷史
        self.training_history = {
            'stage1': {},
            'stage2': {},
            'stage3': {}
        }

    def train_stage1(self, epochs: int = 15) -> Dict[str, Dict]:
        """階段1: 單模態預訓練"""
        print("🚀 開始階段1: 單模態預訓練")
        stage1_history = {}

        for modality in ['visual', 'audio', 'text']:
            print(f"\n📊 訓練 {modality.upper()} 模態...")

            # 創建單模態訓練器
            trainer = SingleModalTrainer(
                modality=modality,
                train_loader=self.data_loaders['train'],
                val_loader=self.data_loaders['val'],
                device=self.device,
                learning_rate=1e-4,
                save_dir=self.save_dir / f"stage1_{modality}"
            )

            # 訓練
            history = trainer.train(epochs=epochs)
            stage1_history[modality] = history

            # 保存模態編碼器
            encoder_path = self.save_dir / f"stage1_{modality}_encoder.pth"
            torch.save(trainer.model.encoders[modality].state_dict(), encoder_path)
            print(f"✅ 保存 {modality} 編碼器: {encoder_path}")

        self.training_history['stage1'] = stage1_history
        return stage1_history

    def train_stage2(self, epochs: int = 20) -> Dict[str, Dict]:
        """階段2: 雙模態對齊學習"""
        print("\n🔥 開始階段2: 雙模態對齊學習")
        stage2_history = {}

        modal_pairs = [
            ['visual', 'audio'],
            ['visual', 'text'],
            ['audio', 'text']
        ]

        for pair in modal_pairs:
            pair_name = f"{pair[0]}_{pair[1]}"
            print(f"\n📊 訓練 {pair[0].upper()} + {pair[1].upper()} 融合...")

            # 創建雙模態訓練器
            trainer = DualModalTrainer(
                modality_pair=pair,
                train_loader=self.data_loaders['train'],
                val_loader=self.data_loaders['val'],
                device=self.device,
                learning_rate=5e-5,
                contrastive_weight=0.1,
                save_dir=self.save_dir / f"stage2_{pair_name}"
            )

            # 載入預訓練編碼器
            for modality in pair:
                encoder_path = self.save_dir / f"stage1_{modality}_encoder.pth"
                if encoder_path.exists():
                    state_dict = torch.load(encoder_path, map_location=self.device)
                    trainer.model.encoders[modality].load_state_dict(state_dict)
                    print(f"✅ 載入預訓練 {modality} 編碼器")

            # 訓練
            history = trainer.train(epochs=epochs)
            stage2_history[pair_name] = history

        self.training_history['stage2'] = stage2_history
        return stage2_history

    def train_stage3(self, epochs: int = 25) -> Dict[str, List[float]]:
        """階段3: 三模態融合訓練"""
        print("\n⚡ 開始階段3: 三模態融合訓練")

        # 創建三模態訓練器
        trainer = TriModalTrainer(
            train_loader=self.data_loaders['train'],
            val_loader=self.data_loaders['val'],
            device=self.device,
            learning_rate=1e-5,
            contrastive_weight=0.1,
            consistency_weight=0.05,
            save_dir=self.save_dir / "stage3_trimodal"
        )

        # 載入預訓練編碼器
        for modality in ['visual', 'audio', 'text']:
            encoder_path = self.save_dir / f"stage1_{modality}_encoder.pth"
            if encoder_path.exists():
                state_dict = torch.load(encoder_path, map_location=self.device)
                trainer.model.encoders[modality].load_state_dict(state_dict)
                print(f"✅ 載入預訓練 {modality} 編碼器")

        # 訓練
        history = trainer.train(epochs=epochs)

        # 保存最終模型
        final_model_path = self.save_dir / "final_trimodal_model.pth"
        torch.save(trainer.model.state_dict(), final_model_path)
        print(f"🏆 保存最終模型: {final_model_path}")

        self.training_history['stage3'] = history
        return history

    def full_training_pipeline(
        self,
        stage1_epochs: int = 15,
        stage2_epochs: int = 20,
        stage3_epochs: int = 25
    ) -> Dict:
        """完整的三階段訓練流程"""
        print("🚀 開始完整三階段漸進式訓練...")

        # 記錄開始時間
        start_time = time.time()

        # 階段1: 單模態預訓練
        stage1_history = self.train_stage1(epochs=stage1_epochs)

        # 階段2: 雙模態對齊學習
        stage2_history = self.train_stage2(epochs=stage2_epochs)

        # 階段3: 三模態融合訓練
        stage3_history = self.train_stage3(epochs=stage3_epochs)

        # 總訓練時間
        total_time = time.time() - start_time

        # 保存訓練歷史
        history_file = self.save_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n🎉 完整訓練完成！總時間: {total_time/3600:.1f} 小時")
        print(f"📊 訓練歷史保存至: {history_file}")

        return self.training_history


def test_training_pipeline():
    """測試訓練流程"""
    print("🧪 測試訓練流程...")

    # 這裡只是展示如何使用，實際需要真實的數據載入器
    from data import create_data_loaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 創建數據載入器
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=8,
            num_workers=0
        )

        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        # 創建漸進式訓練器
        progressive_trainer = ProgressiveTrainer(
            data_loaders=data_loaders,
            device=device,
            save_dir="test_checkpoints"
        )

        print("✅ 訓練器創建成功")
        print("💡 要開始完整訓練，運行: progressive_trainer.full_training_pipeline()")

    except Exception as e:
        print(f"❌ 訓練流程測試失敗: {e}")


if __name__ == "__main__":
    test_training_pipeline()