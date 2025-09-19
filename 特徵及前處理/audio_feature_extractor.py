#!/usr/bin/env python3
"""
詞彙語音特徵提取器 - Multi_SignView 多模態手語辨識系統
專為30個手語詞彙對應的英文單詞語音進行特徵提取
支援多樣化語音下載、MFCC/Spectral特徵提取、多模態融合

技術特點：
- 自動下載30個英文單詞的多樣化語音音檔（不同口音、語速、性別）
- MFCC(13) + Spectral(7) + Temporal(4) = 24維語音特徵
- 支援 TTS 語音生成和線上語音資源下載
- 語音品質評估和篩選機制
- 多語音版本特徵融合策略

詞彙列表：
again, bird, book, computer, cousin, deaf, drink, eat, finish, fish,
friend, good, happy, learn, like, mother, need, nice, no, orange,
school, sister, student, table, teacher, tired, want, what, white, yes

Author: Claude Code + Multi_SignView Team
Date: 2024
"""

import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from tqdm import tqdm
import json
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
import hashlib

# 音訊處理核心依賴
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.stats import zscore

# TTS 和語音下載依賴
import requests
import urllib.request
from urllib.parse import quote
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import shutil

# 限制多執行緒，避免與多進程互搶 CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 專案路徑配置
PROJECT_ROOT = Path(__file__).parent
AUDIO_DATA_ROOT = PROJECT_ROOT / "vocabulary_audio"  # 詞彙語音存放目錄
OUTPUT_ROOT = PROJECT_ROOT / "features" / "audio_features"
NUM_WORKERS = 2  # 保守設定，避免資源競爭
TARGET_SAMPLE_RATE = 16000  # 標準化採樣率

# 30個手語詞彙列表
SIGN_LANGUAGE_VOCABULARY = [
    'again', 'bird', 'book', 'computer', 'cousin', 'deaf', 'drink', 'eat',
    'finish', 'fish', 'friend', 'good', 'happy', 'learn', 'like', 'mother',
    'need', 'nice', 'no', 'orange', 'school', 'sister', 'student', 'table',
    'teacher', 'tired', 'want', 'what', 'white', 'yes'
]

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VocabularyAudioDownloader:
    """
    詞彙語音下載器

    支援多種語音來源的自動下載和管理
    """

    def __init__(self, vocabulary: List[str] = SIGN_LANGUAGE_VOCABULARY,
                 audio_root: Path = AUDIO_DATA_ROOT):
        """
        初始化語音下載器

        Args:
            vocabulary: 詞彙列表
            audio_root: 語音存放根目錄
        """
        self.vocabulary = vocabulary
        self.audio_root = Path(audio_root)
        self.audio_root.mkdir(parents=True, exist_ok=True)

        # TTS 配置
        self.tts_languages = {
            'en-us': 'American English',
            'en-gb': 'British English',
            'en-au': 'Australian English',
            'en-ca': 'Canadian English'
        }

        print(f"🎤 VocabularyAudioDownloader 初始化完成")
        print(f"   - 詞彙數量: {len(vocabulary)}")
        print(f"   - 語音存放目錄: {self.audio_root}")

    def download_tts_audio(self, word: str, lang: str = 'en',
                          tld: str = 'com', slow: bool = False) -> Optional[Path]:
        """
        使用 gTTS 下載單詞語音

        Args:
            word: 單詞
            lang: 語言代碼
            tld: 頂級域名（影響口音）
            slow: 是否慢速發音

        Returns:
            語音檔案路徑或 None
        """
        try:
            # 建立詞彙目錄
            word_dir = self.audio_root / word
            word_dir.mkdir(exist_ok=True)

            # 檔案命名
            speed_suffix = "_slow" if slow else "_normal"
            filename = f"{word}_{lang}_{tld}{speed_suffix}.wav"
            output_path = word_dir / filename

            if output_path.exists():
                return output_path

            # 生成 TTS 語音
            tts = gTTS(text=word, lang=lang, tld=tld, slow=slow)

            # 暫存 mp3 檔案
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tts.save(tmp_file.name)

                # 轉換為 wav 格式
                audio = AudioSegment.from_mp3(tmp_file.name)
                audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
                audio = audio.set_channels(1)  # 單聲道
                audio.export(str(output_path), format="wav")

                # 清理暫存檔
                os.unlink(tmp_file.name)

            print(f"✅ TTS 語音下載完成: {filename}")
            return output_path

        except Exception as e:
            print(f"❌ TTS 語音下載失敗 {word} ({lang}_{tld}): {str(e)}")
            return None

    def download_forvo_audio(self, word: str) -> List[Path]:
        """
        從 Forvo 下載真人發音（需要網路連線）

        Args:
            word: 單詞

        Returns:
            下載的語音檔案路徑列表
        """
        try:
            word_dir = self.audio_root / word
            word_dir.mkdir(exist_ok=True)

            # Forvo API 或網頁爬取（簡化版本）
            # 這裡實作基本的下載邏輯
            downloaded_files = []

            # 模擬不同發音版本的下載
            for i, accent in enumerate(['us', 'uk', 'au']):
                filename = f"{word}_forvo_{accent}.wav"
                output_path = word_dir / filename

                if not output_path.exists():
                    # 這裡應該實作真正的 Forvo API 呼叫
                    # 暫時跳過
                    continue

                downloaded_files.append(output_path)

            return downloaded_files

        except Exception as e:
            print(f"❌ Forvo 語音下載失敗 {word}: {str(e)}")
            return []

    def download_all_vocabulary_audio(self, max_versions_per_word: int = 4) -> Dict[str, List[Path]]:
        """
        下載所有詞彙的多樣化語音

        Args:
            max_versions_per_word: 每個單詞最多下載的語音版本數

        Returns:
            {詞彙: [語音檔案路徑列表]} 字典
        """
        print(f"🎤 開始下載 {len(self.vocabulary)} 個詞彙的語音...")

        vocabulary_audio = {}

        for word in tqdm(self.vocabulary, desc="下載詞彙語音"):
            audio_files = []

            # 1. 下載 gTTS 多種口音版本
            tts_configs = [
                ('en', 'com', False),    # 美式英語，正常語速
                ('en', 'co.uk', False),  # 英式英語，正常語速
                ('en', 'com.au', False), # 澳式英語，正常語速
                ('en', 'com', True),     # 美式英語，慢速
            ]

            for lang, tld, slow in tts_configs[:max_versions_per_word]:
                audio_file = self.download_tts_audio(word, lang, tld, slow)
                if audio_file:
                    audio_files.append(audio_file)

            # 2. 嘗試下載 Forvo 真人發音（如果可用）
            # forvo_files = self.download_forvo_audio(word)
            # audio_files.extend(forvo_files)

            vocabulary_audio[word] = audio_files

            if audio_files:
                print(f"   📁 {word}: {len(audio_files)} 個語音版本")
            else:
                print(f"   ❌ {word}: 沒有下載到語音")

        total_files = sum(len(files) for files in vocabulary_audio.values())
        print(f"🎤 語音下載完成！總共 {total_files} 個語音檔案")

        return vocabulary_audio

    def evaluate_audio_quality(self, audio_path: Path) -> Dict[str, float]:
        """
        評估語音檔案品質

        Args:
            audio_path: 語音檔案路徑

        Returns:
            品質評估指標字典
        """
        try:
            audio, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE)

            # 品質評估指標
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr)[0])

            quality_metrics = {
                'duration': duration,
                'rms_energy': rms_energy,
                'zero_crossing_rate': zcr,
                'spectral_centroid': spectral_centroid,
                'snr_estimate': rms_energy / (zcr + 1e-8)  # 簡單的信噪比估計
            }

            return quality_metrics

        except Exception as e:
            print(f"❌ 語音品質評估失敗 {audio_path}: {str(e)}")
            return {}


class AudioFeatureExtractor:
    """
    多維音訊特徵提取器

    整合 MFCC、Chroma、Spectral、Temporal 四大類音訊特徵
    支援時序對齊、批次處理、多進程加速
    """

    def __init__(self,
                 sample_rate: int = TARGET_SAMPLE_RATE,
                 n_mfcc: int = 13,
                 window_size: int = 2048,
                 hop_length: int = 512):
        """
        初始化音訊特徵提取器

        Args:
            sample_rate: 目標採樣率 (Hz)
            n_mfcc: MFCC 係數數量
            window_size: FFT 窗口大小
            hop_length: 跳躍長度
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.hop_length = hop_length

        # 特徵維度配置（移除 Chroma，專注於語音特徵）
        self.feature_dims = {
            'mfcc': n_mfcc,           # 13維 MFCC 係數
            'spectral': 7,            # 7維 Spectral 特徵
            'temporal': 4             # 4維 Temporal 特徵
        }
        self.total_dims = sum(self.feature_dims.values())  # 總計24維

        print(f"🎵 AudioFeatureExtractor 初始化完成")
        print(f"   - 採樣率: {self.sample_rate} Hz")
        print(f"   - 特徵維度: MFCC({n_mfcc}) + Spectral(7) + Temporal(4) = {self.total_dims}維")
        print(f"   - 專為詞彙語音特徵提取優化")

    def load_and_preprocess_audio(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        載入並預處理詞彙語音檔案

        Args:
            audio_path: 語音檔案路徑

        Returns:
            預處理後的音訊信號 (numpy array) 或 None
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                print(f"❌ 語音檔案不存在: {audio_path}")
                return None

            # 使用 librosa 載入語音檔案
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

            if len(audio) == 0:
                print(f"⚠️  音訊信號為空: {audio_path.name}")
                return None

            # 音訊預處理
            audio = self._preprocess_audio(audio)

            return audio

        except Exception as e:
            print(f"❌ 音訊載入失敗 {audio_path.name}: {str(e)}")
            return None

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        音訊預處理管線

        Args:
            audio: 原始音訊信號

        Returns:
            預處理後的音訊信號
        """
        # 1. 移除 DC 分量
        audio = audio - np.mean(audio)

        # 2. 音量標準化 (RMS 歸一化) - 語音優化
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.3  # 語音特徵優化

        # 3. 低通濾波器 (移除高頻噪音)
        nyquist = self.sample_rate // 2
        cutoff = min(8000, nyquist - 1)  # 8kHz 低通
        b, a = butter(4, cutoff/nyquist, btype='low')
        audio = filtfilt(b, a, audio)

        # 4. 異常值處理
        audio = np.clip(audio, -3*np.std(audio), 3*np.std(audio))

        return audio

    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取 MFCC (Mel-frequency cepstral coefficients) 特徵

        Args:
            audio: 音訊信號

        Returns:
            MFCC 特徵字典
        """
        try:
            # 計算 MFCC 係數
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window='hann'
            )

            # 計算 MFCC 一階和二階差分 (Delta 和 Delta-Delta)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            return {
                'mfcc': mfcc,                    # 基本 MFCC 係數
                'mfcc_delta': mfcc_delta,        # 一階差分
                'mfcc_delta2': mfcc_delta2,      # 二階差分
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                'mfcc_min': np.min(mfcc, axis=1),
                'mfcc_max': np.max(mfcc, axis=1)
            }

        except Exception as e:
            print(f"❌ MFCC 特徵提取失敗: {str(e)}")
            return {}

    def extract_phonetic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取語音學 (Phonetic) 特徵

        Args:
            audio: 音訊信號

        Returns:
            語音學特徵字典
        """
        try:
            # 基頻 (F0) 提取
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            # 清理 NaN 值
            f0_clean = f0[~np.isnan(f0)]

            # 語音持續時間
            voiced_duration = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0.0

            # 語音強度變化
            rms_frames = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            intensity_variation = np.std(rms_frames) if len(rms_frames) > 0 else 0.0

            return {
                'f0_mean': np.mean(f0_clean) if len(f0_clean) > 0 else 0.0,
                'f0_std': np.std(f0_clean) if len(f0_clean) > 0 else 0.0,
                'f0_range': np.ptp(f0_clean) if len(f0_clean) > 0 else 0.0,
                'voiced_duration_ratio': float(voiced_duration),
                'intensity_variation': float(intensity_variation)
            }

        except Exception as e:
            print(f"❌ 語音學特徵提取失敗: {str(e)}")
            return {}

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取頻譜 (Spectral) 特徵

        Args:
            audio: 音訊信號

        Returns:
            Spectral 特徵字典
        """
        try:
            # 1. Spectral Centroid (頻譜重心)
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            # 2. Spectral Bandwidth (頻譜帶寬)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            # 3. Spectral Rolloff (頻譜滾降點)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]

            # 4. Zero Crossing Rate (過零率)
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]

            # 5. Spectral Contrast (頻譜對比度)
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )

            return {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zcr_mean': np.mean(zcr),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'spectral_contrast_mean': np.mean(spectral_contrast)
            }

        except Exception as e:
            print(f"❌ Spectral 特徵提取失敗: {str(e)}")
            return {}

    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取時域 (Temporal) 特徵

        Args:
            audio: 音訊信號

        Returns:
            Temporal 特徵字典
        """
        try:
            # 1. RMS Energy (均方根能量)
            rms_energy = librosa.feature.rms(
                y=audio, hop_length=self.hop_length
            )[0]

            # 2. Tempo 檢測 (節拍)
            try:
                tempo, beats = librosa.beat.beat_track(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                )
            except:
                tempo = 0.0
                beats = []

            # 3. Onset Detection (起始點檢測)
            try:
                onset_frames = librosa.onset.onset_detect(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                )
                onset_rate = len(onset_frames) / (len(audio) / self.sample_rate)
            except:
                onset_rate = 0.0

            # 4. Spectral Flatness (頻譜平坦度)
            spectral_flatness = librosa.feature.spectral_flatness(
                y=audio, hop_length=self.hop_length
            )[0]

            return {
                'rms_energy_mean': np.mean(rms_energy),
                'tempo': float(tempo) if tempo is not None else 0.0,
                'onset_rate': onset_rate,
                'spectral_flatness_mean': np.mean(spectral_flatness)
            }

        except Exception as e:
            print(f"❌ Temporal 特徵提取失敗: {str(e)}")
            return {}

    def extract_audio_features(self, audio_path: Union[str, Path], word: str = None) -> Optional[Dict]:
        """
        從詞彙語音檔案中提取完整音訊特徵

        Args:
            audio_path: 語音檔案路徑
            word: 對應的單詞 (可選)

        Returns:
            完整音訊特徵字典或 None
        """
        try:
            audio_path = Path(audio_path)

            # 從檔案名推斷單詞（如果未提供）
            if word is None:
                word = audio_path.stem.split('_')[0]  # 假設檔名格式為 word_*

            # 載入和預處理音訊
            audio = self.load_and_preprocess_audio(audio_path)
            if audio is None:
                return None

            # 提取各類特徵
            mfcc_features = self.extract_mfcc_features(audio)
            phonetic_features = self.extract_phonetic_features(audio)  # 取代 chroma
            spectral_features = self.extract_spectral_features(audio)
            temporal_features = self.extract_temporal_features(audio)

            # 整合所有特徵
            features = {
                'audio_path': str(audio_path),
                'word': word,
                'audio_duration': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'mfcc': mfcc_features,
                'phonetic': phonetic_features,
                'spectral': spectral_features,
                'temporal': temporal_features,
                'feature_dims': self.feature_dims,
                'total_dims': self.total_dims
            }

            return features

        except Exception as e:
            print(f"❌ 音訊特徵提取失敗 {audio_path.name}: {str(e)}")
            return None

    def _extract_temporal_aligned_features(self, audio: np.ndarray) -> Dict[str, List]:
        """
        提取與100幀影像對齊的時序音訊特徵

        Args:
            audio: 音訊信號

        Returns:
            時序對齊的特徵序列
        """
        try:
            # 計算音訊總時長
            duration = len(audio) / self.sample_rate

            # 分割為100個時間片段
            frame_duration = duration / self.target_frames
            frame_samples = int(frame_duration * self.sample_rate)

            aligned_features = {
                'frame_energy': [],
                'frame_zcr': [],
                'frame_spectral_centroid': [],
                'frame_spectral_bandwidth': []
            }

            for frame_idx in range(self.target_frames):
                start_sample = int(frame_idx * frame_samples)
                end_sample = min(start_sample + frame_samples, len(audio))

                if start_sample >= len(audio):
                    # 處理音訊長度不足的情況，填充零值
                    aligned_features['frame_energy'].append(0.0)
                    aligned_features['frame_zcr'].append(0.0)
                    aligned_features['frame_spectral_centroid'].append(0.0)
                    aligned_features['frame_spectral_bandwidth'].append(0.0)
                else:
                    frame_audio = audio[start_sample:end_sample]

                    # 計算該幀的音訊特徵
                    if len(frame_audio) > 0:
                        rms = np.sqrt(np.mean(frame_audio**2))
                        zcr = np.mean(librosa.feature.zero_crossing_rate(frame_audio)[0])

                        # 頻譜特徵（需要足夠的樣本）
                        if len(frame_audio) >= 512:
                            spec_cent = np.mean(librosa.feature.spectral_centroid(
                                y=frame_audio, sr=self.sample_rate)[0])
                            spec_bw = np.mean(librosa.feature.spectral_bandwidth(
                                y=frame_audio, sr=self.sample_rate)[0])
                        else:
                            spec_cent = 0.0
                            spec_bw = 0.0

                        aligned_features['frame_energy'].append(float(rms))
                        aligned_features['frame_zcr'].append(float(zcr))
                        aligned_features['frame_spectral_centroid'].append(float(spec_cent))
                        aligned_features['frame_spectral_bandwidth'].append(float(spec_bw))
                    else:
                        aligned_features['frame_energy'].append(0.0)
                        aligned_features['frame_zcr'].append(0.0)
                        aligned_features['frame_spectral_centroid'].append(0.0)
                        aligned_features['frame_spectral_bandwidth'].append(0.0)

            return aligned_features

        except Exception as e:
            print(f"❌ 時序對齊特徵提取失敗: {str(e)}")
            return {}

    def extract_normalized_features(self, features: Dict) -> np.ndarray:
        """
        提取標準化的特徵向量 (24維)

        Args:
            features: 完整特徵字典

        Returns:
            24維標準化特徵向量
        """
        try:
            feature_vector = []

            # 1. MFCC 特徵 (13維)
            if 'mfcc' in features and 'mfcc_mean' in features['mfcc']:
                mfcc_mean = features['mfcc']['mfcc_mean'][:self.n_mfcc]
                feature_vector.extend(mfcc_mean)
            else:
                feature_vector.extend([0.0] * self.n_mfcc)

            # 2. Spectral 特徵 (7維)
            spectral_keys = [
                'spectral_centroid_mean', 'spectral_bandwidth_mean',
                'spectral_rolloff_mean', 'zcr_mean',
                'spectral_centroid_std', 'spectral_bandwidth_std',
                'spectral_contrast_mean'
            ]
            for key in spectral_keys:
                if 'spectral' in features and key in features['spectral']:
                    feature_vector.append(features['spectral'][key])
                else:
                    feature_vector.append(0.0)

            # 3. Temporal 特徵 (4維)
            temporal_keys = [
                'rms_energy_mean', 'tempo', 'onset_rate', 'spectral_flatness_mean'
            ]
            for key in temporal_keys:
                if 'temporal' in features and key in features['temporal']:
                    feature_vector.append(features['temporal'][key])
                else:
                    feature_vector.append(0.0)

            # 轉換為 numpy 陣列並標準化
            feature_vector = np.array(feature_vector, dtype=np.float32)

            # Z-score 標準化
            if np.std(feature_vector) > 0:
                feature_vector = zscore(feature_vector)

            # 處理 NaN 和 Inf
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)

            return feature_vector

        except Exception as e:
            print(f"❌ 特徵向量標準化失敗: {str(e)}")
            return np.zeros(self.total_dims, dtype=np.float32)

    def extract_word_features_from_multiple_versions(self, audio_files: List[Path], word: str) -> Dict:
        """
        從多個語音版本中提取融合特徵

        Args:
            audio_files: 同一單詞的多個語音檔案
            word: 單詞

        Returns:
            融合的特徵字典
        """
        try:
            all_features = []
            all_vectors = []

            for audio_file in audio_files:
                features = self.extract_audio_features(audio_file, word)
                if features:
                    all_features.append(features)
                    vector = self.extract_normalized_features(features)
                    all_vectors.append(vector)

            if not all_vectors:
                return {}

            # 特徵融合策略
            all_vectors = np.array(all_vectors)

            # 1. 平均值融合
            mean_vector = np.mean(all_vectors, axis=0)

            # 2. 最大值融合 (保留最顕著的特徵)
            max_vector = np.max(all_vectors, axis=0)

            # 3. 加權融合 (品質高的版本權重更大)
            weights = []
            for features in all_features:
                duration = features.get('audio_duration', 1.0)
                # 簡單品質評估：時長適中的語音品質較好
                quality_score = 1.0 / (1.0 + abs(duration - 1.0))  # 假設 1 秒是理想時長
                weights.append(quality_score)

            weights = np.array(weights)
            weights = weights / np.sum(weights)  # 歸一化

            weighted_vector = np.average(all_vectors, axis=0, weights=weights)

            # 返回融合結果
            fusion_result = {
                'word': word,
                'num_versions': len(all_features),
                'version_details': [{
                    'file': str(f['audio_path']),
                    'duration': f['audio_duration'],
                    'quality_weight': w
                } for f, w in zip(all_features, weights)],
                'features': {
                    'mean_fusion': mean_vector,
                    'max_fusion': max_vector,
                    'weighted_fusion': weighted_vector,
                    'std_across_versions': np.std(all_vectors, axis=0)
                },
                'recommended_vector': weighted_vector,  # 推薦使用加權融合
                'feature_dims': self.feature_dims,
                'total_dims': self.total_dims
            }

            return fusion_result

        except Exception as e:
            print(f"❌ 多版本特徵融合失敗 {word}: {str(e)}")
            return {}


def extract_single_word_audio(word: str, audio_files: List[Path], extractor: AudioFeatureExtractor) -> Tuple[str, Optional[Dict]]:
    """
    單一單詞的語音特徵提取 (多進程處理函數)

    Args:
        word: 單詞
        audio_files: 該單詞的語音檔案列表
        extractor: 音訊特徵提取器實例

    Returns:
        (單詞, 特徵字典) 或 (單詞, None)
    """
    try:
        features = extractor.extract_word_features_from_multiple_versions(audio_files, word)
        return (word, features)
    except Exception as e:
        print(f"❌ 單一單詞處理失敗 {word}: {str(e)}")
        return (word, None)


def run_vocabulary_audio_extraction(
    audio_root: Optional[str] = None,
    output_root: Optional[str] = None,
    num_workers: int = NUM_WORKERS,
    download_audio: bool = True,
    vocabulary: List[str] = SIGN_LANGUAGE_VOCABULARY
) -> Dict[str, int]:
    """
    批次詞彙語音特徵提取主函數

    Args:
        audio_root: 詞彙語音根目錄路徑
        output_root: 輸出根目錄路徑
        num_workers: 多進程工作數量
        download_audio: 是否自動下載語音
        vocabulary: 詞彙列表

    Returns:
        處理結果統計字典
    """
    # 路徑配置
    audio_data_path = Path(audio_root) if audio_root else AUDIO_DATA_ROOT
    output_path = Path(output_root) if output_root else OUTPUT_ROOT

    # 建立輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🎤 開始詞彙語音特徵提取")
    print(f"   📂 語音根目錄: {audio_data_path}")
    print(f"   📂 輸出路徑: {output_path}")
    print(f"   👥 工作進程數: {num_workers}")
    print(f"   📚 詞彙數量: {len(vocabulary)}")

    # 1. 初始化語音下載器和特徵提取器
    if download_audio:
        downloader = VocabularyAudioDownloader(vocabulary, audio_data_path)
        print("🎤 開始下載詞彙語音...")
        vocabulary_audio = downloader.download_all_vocabulary_audio(max_versions_per_word=4)
    else:
        # 直接扫描現有的語音檔案
        vocabulary_audio = {}
        for word in vocabulary:
            word_dir = audio_data_path / word
            if word_dir.exists():
                audio_files = list(word_dir.glob('*.wav')) + list(word_dir.glob('*.mp3'))
                vocabulary_audio[word] = audio_files

    extractor = AudioFeatureExtractor()

    # 統計語音檔案
    total_audio_files = sum(len(files) for files in vocabulary_audio.values())
    print(f"   🎤 找到 {total_audio_files} 個語音檔案")

    # 2. 多進程批次處理
    results = {
        'total_words': len(vocabulary),
        'success_count': 0,
        'failed_count': 0,
        'total_audio_files': total_audio_files,
        'processed_words': set()
    }

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任務
        futures = {
            executor.submit(extract_single_word_audio, word, audio_files, extractor): word
            for word, audio_files in vocabulary_audio.items() if audio_files
        }

        # 處理結果
        for future in tqdm(as_completed(futures), total=len(futures), desc="🎤 詞彙語音特徵提取"):
            word = futures[future]
            try:
                word_name, features = future.result()

                if features and 'recommended_vector' in features:
                    # 儲存完整特徵
                    output_file = output_path / f"{word}_audio_features.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(features, f, indent=2, ensure_ascii=False, default=str)

                    # 儲存推薦的標準化特徵向量
                    recommended_vector = features['recommended_vector']
                    vector_file = output_path / f"{word}_normalized_24d.npy"
                    np.save(vector_file, recommended_vector)

                    # 儲存所有融合版本
                    for fusion_type, vector in features['features'].items():
                        fusion_file = output_path / f"{word}_{fusion_type}_24d.npy"
                        np.save(fusion_file, vector)

                    results['success_count'] += 1
                    results['processed_words'].add(word)

                    print(f"   ✅ {word}: {features['num_versions']}個語音版本融合完成")
                else:
                    results['failed_count'] += 1
                    print(f"   ❌ {word}: 特徵提取失敗")

            except Exception as e:
                print(f"❌ 處理結果失敗 {word}: {str(e)}")
                results['failed_count'] += 1

    # 3. 輸出處理結果
    print(f"🎤 詞彙語音特徵提取完成!")
    print(f"   ✅ 成功: {results['success_count']}/{results['total_words']} 單詞")
    print(f"   ❌ 失敗: {results['failed_count']} 單詞")
    print(f"   🎤 總語音檔案: {results['total_audio_files']}")
    print(f"   📊 成功率: {results['success_count']/results['total_words']*100:.1f}%")
    print(f"   📁 輸出目錄: {output_path}")

    return results


if __name__ == "__main__":
    # 執行詞彙語音特徵提取
    results = run_vocabulary_audio_extraction(download_audio=True)