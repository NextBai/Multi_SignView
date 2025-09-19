#!/usr/bin/env python3
"""
Features 目錄統一整合管理器
負責檢查、驗證和管理所有模態的特徵檔案
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

class FeaturesIntegrationManager:
    """統一管理所有模態特徵的整合器"""

    def __init__(self, features_root: str = "features"):
        self.features_root = Path(features_root)
        self.vocabulary = [
            'again', 'bird', 'book', 'computer', 'cousin', 'deaf', 'drink', 'eat',
            'finish', 'fish', 'friend', 'good', 'happy', 'learn', 'like', 'mother',
            'need', 'nice', 'no', 'orange', 'school', 'sister', 'student', 'table',
            'teacher', 'tired', 'want', 'what', 'white', 'yes'
        ]

        # 特徵目錄結構
        self.feature_dirs = {
            'audio': self.features_root / 'audio_features',
            'visual_mediapipe': self.features_root / 'mediapipe_features',
            'visual_optical_flow': self.features_root / 'optical_flow_features',
            'text_embeddings': self.features_root / 'text_embeddings',
            'semantic': self.features_root / 'semantic_features'
        }

    def check_directory_structure(self) -> Dict[str, bool]:
        """檢查所有特徵目錄是否存在"""
        structure_status = {}

        print("🔍 檢查 features 目錄結構...")
        for modality, path in self.feature_dirs.items():
            exists = path.exists()
            structure_status[modality] = exists
            status = "✅" if exists else "❌"
            print(f"   {status} {modality}: {path}")

        return structure_status

    def analyze_audio_features(self) -> Dict[str, Dict[str, bool]]:
        """分析音訊特徵完整性"""
        print("\n🎤 分析音訊特徵...")
        audio_status = {}

        audio_types = [
            'audio_features.json',
            'max_fusion_24d.npy',
            'mean_fusion_24d.npy',
            'normalized_24d.npy',
            'std_across_versions_24d.npy',
            'weighted_fusion_24d.npy'
        ]

        for word in self.vocabulary:
            word_status = {}
            for audio_type in audio_types:
                file_path = self.feature_dirs['audio'] / f"{word}_{audio_type}"
                exists = file_path.exists()
                word_status[audio_type] = exists

            audio_status[word] = word_status

        # 統計完整性
        complete_count = sum(1 for word_data in audio_status.values()
                           if all(word_data.values()))
        print(f"   📊 音訊特徵完整度: {complete_count}/{len(self.vocabulary)} 詞彙")

        return audio_status

    def analyze_visual_features(self) -> Dict[str, Dict[str, any]]:
        """分析視覺特徵完整性"""
        print("\n👁️ 分析視覺特徵...")
        visual_status = {}

        # MediaPipe 特徵
        mediapipe_status = {}
        for word in self.vocabulary:
            word_dir = self.feature_dirs['visual_mediapipe'] / word
            if word_dir.exists():
                npy_files = list(word_dir.glob("*.npy"))
                mediapipe_status[word] = {
                    'exists': True,
                    'file_count': len(npy_files),
                    'files': [f.name for f in npy_files[:5]]  # 前5個檔案
                }
            else:
                mediapipe_status[word] = {'exists': False, 'file_count': 0}

        # 光流特徵
        optical_flow_status = {}
        for word in self.vocabulary:
            word_dir = self.feature_dirs['visual_optical_flow'] / word
            if word_dir.exists():
                npy_files = list(word_dir.glob("*.npy"))
                optical_flow_status[word] = {
                    'exists': True,
                    'file_count': len(npy_files)
                }
            else:
                optical_flow_status[word] = {'exists': False, 'file_count': 0}

        visual_status['mediapipe'] = mediapipe_status
        visual_status['optical_flow'] = optical_flow_status

        # 統計
        mp_complete = sum(1 for data in mediapipe_status.values() if data['exists'])
        of_complete = sum(1 for data in optical_flow_status.values() if data['exists'])

        print(f"   📊 MediaPipe 特徵: {mp_complete}/{len(self.vocabulary)} 詞彙")
        print(f"   📊 光流特徵: {of_complete}/{len(self.vocabulary)} 詞彙")

        return visual_status

    def analyze_text_features(self) -> Dict[str, bool]:
        """分析文字特徵完整性"""
        print("\n📝 分析文字特徵...")
        text_status = {}

        expected_files = [
            'word2vec_embeddings.npy',
            'fasttext_embeddings.npy',
            'bert_embeddings.npy',
            'unified_embeddings.npy'
        ]

        for file_name in expected_files:
            file_path = self.feature_dirs['text_embeddings'] / file_name
            exists = file_path.exists()
            text_status[file_name] = exists

            if exists:
                try:
                    data = np.load(file_path)
                    print(f"   ✅ {file_name}: {data.shape}")
                except Exception as e:
                    print(f"   ❌ {file_name}: 載入失敗 - {e}")
            else:
                print(f"   ❌ {file_name}: 檔案不存在")

        return text_status

    def analyze_semantic_features(self) -> Dict[str, bool]:
        """分析語義特徵完整性"""
        print("\n🧠 分析語義特徵...")
        semantic_status = {}

        expected_files = [
            'semantic_analysis.json',
            'semantic_analysis_fixed.json',
            'semantic_analysis.json.backup'
        ]

        for file_name in expected_files:
            file_path = self.feature_dirs['semantic'] / file_name
            exists = file_path.exists()
            semantic_status[file_name] = exists

            if exists and file_name.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"   ✅ {file_name}: 包含 {len(data)} 個鍵")
                except Exception as e:
                    print(f"   ❌ {file_name}: JSON 解析失敗 - {e}")
            else:
                status = "✅" if exists else "❌"
                print(f"   {status} {file_name}")

        return semantic_status

    def generate_integration_report(self) -> Dict:
        """生成完整的整合報告"""
        print("🔧 生成 Features 整合報告...")
        print("=" * 60)

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'vocabulary_count': len(self.vocabulary),
            'vocabulary': self.vocabulary
        }

        # 檢查目錄結構
        report['directory_structure'] = self.check_directory_structure()

        # 分析各模態特徵
        report['audio_features'] = self.analyze_audio_features()
        report['visual_features'] = self.analyze_visual_features()
        report['text_features'] = self.analyze_text_features()
        report['semantic_features'] = self.analyze_semantic_features()

        return report

    def create_trimodal_mapping(self) -> Dict[str, Dict]:
        """創建三模態對應關係"""
        print("\n🔗 創建三模態對應關係...")

        trimodal_mapping = {}

        for word in self.vocabulary:
            mapping = {
                'word': word,
                'modalities': {
                    'visual': {
                        'mediapipe_dir': str(self.feature_dirs['visual_mediapipe'] / word),
                        'optical_flow_dir': str(self.feature_dirs['visual_optical_flow'] / word),
                        'available': (self.feature_dirs['visual_mediapipe'] / word).exists()
                    },
                    'audio': {
                        'features_json': str(self.feature_dirs['audio'] / f"{word}_audio_features.json"),
                        'fusion_variants': [
                            str(self.feature_dirs['audio'] / f"{word}_max_fusion_24d.npy"),
                            str(self.feature_dirs['audio'] / f"{word}_mean_fusion_24d.npy"),
                            str(self.feature_dirs['audio'] / f"{word}_normalized_24d.npy"),
                            str(self.feature_dirs['audio'] / f"{word}_weighted_fusion_24d.npy")
                        ],
                        'available': (self.feature_dirs['audio'] / f"{word}_audio_features.json").exists()
                    },
                    'text': {
                        'embeddings_dir': str(self.feature_dirs['text_embeddings']),
                        'semantic_file': str(self.feature_dirs['semantic'] / 'semantic_analysis_fixed.json'),
                        'available': (self.feature_dirs['text_embeddings'] / 'unified_embeddings.npy').exists()
                    }
                },
                'cross_modal_links': {
                    'visual_audio_alignment': f"video_{word} ↔ audio_{word}",
                    'visual_text_mapping': f"video_{word} ↔ text_{word}",
                    'audio_text_correspondence': f"audio_{word} ↔ text_{word}",
                    'trimodal_fusion': f"video_{word} + audio_{word} + text_{word}"
                }
            }

            trimodal_mapping[word] = mapping

        return trimodal_mapping

    def save_integration_results(self, report: Dict, mapping: Dict) -> None:
        """保存整合結果"""
        # 保存報告
        report_path = self.features_root / 'integration_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 整合報告已保存: {report_path}")

        # 保存三模態對應關係
        mapping_path = self.features_root / 'trimodal_mapping.json'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"💾 三模態對應關係已保存: {mapping_path}")

def main():
    """主函數"""
    print("🚀 Features 目錄統一整合管理器")
    print("=" * 60)

    # 初始化管理器
    manager = FeaturesIntegrationManager()

    # 生成整合報告
    report = manager.generate_integration_report()

    # 創建三模態對應關係
    trimodal_mapping = manager.create_trimodal_mapping()

    # 保存結果
    manager.save_integration_results(report, trimodal_mapping)

    print("\n✅ Features 整合管理完成!")
    print("   📊 可查看 features/integration_report.json")
    print("   🔗 可查看 features/trimodal_mapping.json")

if __name__ == "__main__":
    main()