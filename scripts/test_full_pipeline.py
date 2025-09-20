#!/usr/bin/env python3
"""
Complete Pipeline Test Script
完整流程測試腳本

測試從數據載入到模型訓練的完整流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

# 導入自定義模組
from data import create_data_loaders, TriModalDataset
from models import MultiModalSignClassifier, create_multimodal_classifier
from training import ProgressiveTrainer
from training.utils import set_random_seed, get_device, ModelSummary


def test_data_loading():
    """測試數據載入"""
    print("\n🔍 1. 測試數據載入...")

    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=8,
            num_workers=0,  # 測試時使用單線程
            modalities=['visual', 'audio', 'text']
        )

        print("✅ 數據載入器創建成功")

        # 測試一個批次
        for i, (features, labels) in enumerate(train_loader):
            print(f"✅ 批次 {i+1}:")
            print(f"   標籤形狀: {labels.shape}")
            for modality, tensor in features.items():
                print(f"   {modality:>6}: {tensor.shape}")

            if i >= 1:  # 只測試前2個批次
                break

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        return None, None, None


def test_model_creation():
    """測試模型創建"""
    print("\n🧠 2. 測試模型創建...")

    try:
        # 測試不同模態組合
        modal_configs = [
            (['visual'], "單模態-視覺"),
            (['audio'], "單模態-音訊"),
            (['text'], "單模態-文字"),
            (['visual', 'audio'], "雙模態-視覺+音訊"),
            (['visual', 'text'], "雙模態-視覺+文字"),
            (['audio', 'text'], "雙模態-音訊+文字"),
            (['visual', 'audio', 'text'], "三模態-完整")
        ]

        for modalities, desc in modal_configs:
            model = create_multimodal_classifier(
                num_classes=30,
                modalities=modalities,
                fusion_strategy='attention'
            )

            info = model.get_model_info()
            print(f"✅ {desc}: {info['total_parameters']:,} 參數")

        print("✅ 所有模型配置創建成功")
        return modal_configs[-1][0]  # 返回三模態配置

    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        return None


def test_model_inference():
    """測試模型推理"""
    print("\n⚡ 3. 測試模型推理...")

    try:
        device = get_device(prefer_cuda=False)  # 測試時使用CPU

        # 創建測試模型
        model = create_multimodal_classifier(
            num_classes=30,
            modalities=['visual', 'audio', 'text']
        ).to(device)

        # 創建模擬數據
        batch_size = 4
        test_features = {
            'visual': torch.randn(batch_size, 100, 417).to(device),
            'audio': torch.randn(batch_size, 24).to(device),
            'text': torch.randn(batch_size, 300).to(device)
        }

        # 測試不同推理模式
        model.eval()
        with torch.no_grad():
            # 基礎推理
            logits = model(test_features)
            print(f"✅ 基礎推理輸出: {logits.shape}")

            # 帶嵌入的推理
            logits, embeddings = model(test_features, return_embeddings=True)
            print(f"✅ 帶嵌入推理: logits {logits.shape}, 嵌入數量 {len(embeddings)}")

            # 完整推理（含注意力權重）
            logits, embeddings, attention_weights = model(
                test_features,
                return_embeddings=True,
                return_attention_weights=True
            )
            print(f"✅ 完整推理: 注意力權重數量 {len(attention_weights)}")

        print("✅ 模型推理測試成功")
        return True

    except Exception as e:
        print(f"❌ 模型推理失敗: {e}")
        return False


def test_training_setup():
    """測試訓練設置"""
    print("\n🏋️ 4. 測試訓練設置...")

    try:
        # 設置隨機種子
        set_random_seed(42)

        # 獲取設備
        device = get_device(prefer_cuda=False)

        # 創建數據載入器
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=4,  # 小批次測試
            num_workers=0
        )

        if train_loader is None:
            print("❌ 無法創建數據載入器")
            return False

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
        print("✅ 訓練設置測試完成")

        return progressive_trainer

    except Exception as e:
        print(f"❌ 訓練設置失敗: {e}")
        return None


def test_single_epoch_training():
    """測試單個epoch訓練"""
    print("\n🎯 5. 測試單個epoch訓練...")

    try:
        # 創建簡化的訓練器
        from training.trainer import SingleModalTrainer

        device = get_device(prefer_cuda=False)

        train_loader, val_loader, _ = create_data_loaders(
            batch_size=4,
            num_workers=0
        )

        if train_loader is None:
            print("❌ 數據載入器創建失敗")
            return False

        # 測試單模態訓練（視覺）
        trainer = SingleModalTrainer(
            modality='visual',
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=1e-3,  # 較高學習率加快測試
            save_dir="test_checkpoints/single_modal"
        )

        print("🚀 開始單個epoch訓練測試...")

        # 訓練一個epoch
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()

        print(f"✅ 訓練完成:")
        print(f"   訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.3f}")
        print(f"   驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.3f}")

        return True

    except Exception as e:
        print(f"❌ 單epoch訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_system_info():
    """打印系統信息"""
    print("🖥️  系統信息:")
    print(f"   Python版本: {sys.version}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU數量: {torch.cuda.device_count()}")
        print(f"   當前GPU: {torch.cuda.get_device_name()}")

    # 檢查必要文件
    required_files = [
        "features/trimodal_mapping.json",
        "features/text_embeddings/unified_embeddings.npy",
        "features/audio_features",
        "features/mediapipe_features"
    ]

    print(f"\n📂 檔案檢查:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (不存在)")


def main():
    """主測試函數"""
    print("="*60)
    print("🚀 Multi_SignViews 完整流程測試")
    print("="*60)

    # 打印系統信息
    print_system_info()

    # 測試步驟
    test_steps = [
        ("數據載入", test_data_loading),
        ("模型創建", test_model_creation),
        ("模型推理", test_model_inference),
        ("訓練設置", test_training_setup),
        ("單epoch訓練", test_single_epoch_training)
    ]

    results = {}

    for step_name, test_func in test_steps:
        try:
            result = test_func()
            results[step_name] = result is not False and result is not None
        except Exception as e:
            print(f"❌ {step_name} 測試異常: {e}")
            results[step_name] = False

    # 打印測試總結
    print("\n" + "="*60)
    print("📊 測試結果總結")
    print("="*60)

    for step_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{step_name:>15}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\n總體結果: {passed_tests}/{total_tests} 測試通過")

    if passed_tests == total_tests:
        print("🎉 所有測試通過！系統準備就緒。")
        print("\n💡 下一步:")
        print("   1. 運行完整訓練: python scripts/train_full_pipeline.py")
        print("   2. 評估模型性能: python scripts/evaluate_model.py")
    else:
        print("⚠️  部分測試失敗，請檢查配置和依賴。")

    print("="*60)


if __name__ == "__main__":
    main()