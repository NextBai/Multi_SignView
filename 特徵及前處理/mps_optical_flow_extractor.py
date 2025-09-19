#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from tqdm import tqdm
import json
import numpy as np
import zipfile
import random
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path as _PathAlias  # alias to avoid confusion with typing
import hashlib
from pathlib import Path as _P

# 限制多執行緒，避免與多進程互搶 CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

PROJECT_ROOT = _P(__file__).parent
# 本地資料集與輸出路徑（與訓練腳本一致）
DATASET_PATH = PROJECT_ROOT / "bai_datasets"
OUTPUT_ROOT = PROJECT_ROOT / "features" / "optical_flow_features"
# 保守 worker 數，避免本機資源爭用
NUM_WORKERS = 2
TARGET_FRAMES_DEFAULT = 100


class OpticalFlowExtractor:
    def __init__(self, 
                 target_frames: int = 100,
                 flow_method: str = 'farneback',
                 resize_dims: Tuple[int, int] = (224, 224),
                 confidence_threshold: float = 0.5,
                 roi_mode: str = 'dynamic',
                 dynamic_roi_params: Optional[Dict[str, float]] = None):
        """
        光流運動特徵提取器
        
        Args:
            target_frames: 目標幀數（線性插值的統一長度）
            flow_method: 光流計算方法 ('farneback', 'lucas_kanade', 'tvl1')
            resize_dims: 影像縮放尺寸
            confidence_threshold: 置信度閾值
        """
        self.target_frames = target_frames
        self.flow_method = flow_method
        self.resize_dims = resize_dims
        self.confidence_threshold = confidence_threshold
        self.roi_mode = roi_mode  # 'dynamic' or 'static'
        
        # 光流參數配置
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        self.lucas_kanade_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # ROI區域定義 (手語關注區域)
        self.roi_regions = {
            'hands': [(0.1, 0.3, 0.9, 0.9)],  # 手部區域 (x1, y1, x2, y2)
            'upper_body': [(0.2, 0.1, 0.8, 0.7)],  # 上半身
            'face': [(0.3, 0.0, 0.7, 0.4)],  # 面部表情
            'full_frame': [(0.0, 0.0, 1.0, 1.0)]  # 全畫面
        }
        # 動態 ROI 參數與狀態
        self.dynamic_roi_params = dynamic_roi_params or {
            'mag_percentile': 85.0,       # 以幅度分位數決定動作區域
            'min_ratio_pixels': 0.0025,   # 最小像素比例，太少則忽略
            'expand_ratio': 0.15,         # 邊界擴張比例
            'ema_alpha': 0.3,             # ROI 平滑係數
            'min_points': 10              # LK 最少點數門檻
        }
        self._roi_bbox_norm: Optional[Tuple[float, float, float, float]] = None
        
        print(f"✅ OpticalFlowExtractor 初始化完成")
        print(f"🎯 光流方法: {flow_method}")
        print(f"📏 目標尺寸: {resize_dims}")
        print(f"🎬 目標幀數: {target_frames}")
        print(f"📦 ROI 模式: {self.roi_mode}")

    def extract_video_features(self, video_path: str) -> Optional[Dict]:
        """從影片中提取光流特徵"""
        if not os.path.exists(video_path):
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6 or not np.isfinite(fps):
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps else 0.0
        
        features = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'flow_method': self.flow_method,
                'roi_mode': self.roi_mode
            },
            'frames_data': []
        }
        
        # 依目標幀數取樣，串流方式即時計算光流，降低記憶體用量
        step = max(1, int(round(frame_count / float(self.target_frames)))) if frame_count > 0 else 1
        prev_gray: Optional[np.ndarray] = None
        processed_idx = 0
        flow_sequence: List[Dict] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx % step) != 0:
                frame_idx += 1
                continue

            resized_frame = cv2.resize(frame, self.resize_dims)
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # 計算光流特徵（依方法分派）
                if self.flow_method == 'farneback':
                    fb = self.farneback_params
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray_frame, None,
                        fb['pyr_scale'], fb['levels'], fb['winsize'],
                        fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                    )
                    # 若使用動態 ROI，先更新 ROI bbox 再抽特徵
                    if self.roi_mode == 'dynamic':
                        self._update_dynamic_roi_from_flow(flow)
                        flow_features = self._extract_farneback_features(flow, processed_idx - 1, use_dynamic_roi=True)
                    else:
                        flow_features = self._extract_farneback_features(flow, processed_idx - 1)
                elif self.flow_method == 'lucas_kanade':
                    flow_features = self._extract_lucas_kanade_features(prev_gray, gray_frame, processed_idx - 1)
                elif self.flow_method == 'tvl1':
                    try:
                        flow_features = self._extract_tvl1_features(prev_gray, gray_frame, processed_idx - 1)
                    except Exception:
                        # 沒有 contrib 或建立失敗時退回 Farneback
                        fb = self.farneback_params
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray_frame, None,
                            fb['pyr_scale'], fb['levels'], fb['winsize'],
                            fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                        )
                        if self.roi_mode == 'dynamic':
                            self._update_dynamic_roi_from_flow(flow)
                            flow_features = self._extract_farneback_features(flow, processed_idx - 1, use_dynamic_roi=True)
                        else:
                            flow_features = self._extract_farneback_features(flow, processed_idx - 1)
                else:
                    # 預設 Farneback
                    fb = self.farneback_params
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray_frame, None,
                        fb['pyr_scale'], fb['levels'], fb['winsize'],
                        fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                    )
                    if self.roi_mode == 'dynamic':
                        self._update_dynamic_roi_from_flow(flow)
                        flow_features = self._extract_farneback_features(flow, processed_idx - 1, use_dynamic_roi=True)
                    else:
                        flow_features = self._extract_farneback_features(flow, processed_idx - 1)

                if flow_features is not None:
                    flow_sequence.append(flow_features)

            prev_gray = gray_frame
            processed_idx += 1
            frame_idx += 1

        cap.release()

        if len(flow_sequence) < 1:
            return None

        features['frames_data'] = flow_sequence
        
        # 應用線性插值進行時序對齊
        aligned_features = self._apply_temporal_interpolation(features)
        
        return aligned_features

    def _compute_optical_flow_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """計算光流序列（保留以兼容，但預設不使用全量載入）"""
        flow_sequence = []
        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            curr_frame = frames[i + 1]

            if self.flow_method == 'farneback':
                fb = self.farneback_params
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None,
                    fb['pyr_scale'], fb['levels'], fb['winsize'],
                    fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                )
                flow_features = self._extract_farneback_features(flow, i)
            elif self.flow_method == 'lucas_kanade':
                flow_features = self._extract_lucas_kanade_features(prev_frame, curr_frame, i)
            elif self.flow_method == 'tvl1':
                try:
                    flow_features = self._extract_tvl1_features(prev_frame, curr_frame, i)
                except Exception:
                    fb = self.farneback_params
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, curr_frame, None,
                        fb['pyr_scale'], fb['levels'], fb['winsize'],
                        fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                    )
                    flow_features = self._extract_farneback_features(flow, i)
            else:
                fb = self.farneback_params
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None,
                    fb['pyr_scale'], fb['levels'], fb['winsize'],
                    fb['iterations'], fb['poly_n'], fb['poly_sigma'], fb['flags']
                )
                flow_features = self._extract_farneback_features(flow, i)

            if flow_features is not None:
                flow_sequence.append(flow_features)

        return flow_sequence

    def _extract_farneback_features(self, flow: np.ndarray, frame_idx: int, use_dynamic_roi: bool = False) -> Optional[Dict]:
        """提取Farneback光流特徵"""
        if flow is None:
            return None
            
        # 計算光流的幅度和角度
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 提取ROI區域特徵
        roi_features = {}
        
        if use_dynamic_roi and self._roi_bbox_norm is not None:
            # 以動態 ROI 為 hands 區域，其餘沿用靜態
            dyn_roi = self._roi_bbox_norm
            roi_features['hands'] = self._extract_roi_flow_features(magnitude, angle, dyn_roi)
            for roi_name, roi_coords in self.roi_regions.items():
                if roi_name == 'hands':
                    continue
                roi_data = self._extract_roi_flow_features(magnitude, angle, roi_coords[0])
                roi_features[roi_name] = roi_data
        else:
            for roi_name, roi_coords in self.roi_regions.items():
                roi_data = self._extract_roi_flow_features(magnitude, angle, roi_coords[0])
                roi_features[roi_name] = roi_data
        
        # 全局統計特徵
        global_features = {
            'mean_magnitude': float(np.mean(magnitude)),
            'std_magnitude': float(np.std(magnitude)),
            'max_magnitude': float(np.max(magnitude)),
            'mean_angle': float(np.mean(angle)),
            'std_angle': float(np.std(angle))
        }
        
        return {
            'frame_idx': frame_idx,
            'flow_method': 'farneback',
            'roi_features': roi_features,
            'global_features': global_features,
            'flow_shape': flow.shape[:2]
        }

    def _update_dynamic_roi_from_flow(self, flow: np.ndarray):
        """根據當前光流更新動態 ROI（不依賴關鍵點）"""
        if flow is None or flow.size == 0:
            return
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h, w = mag.shape
        if h == 0 or w == 0:
            return

        params = self.dynamic_roi_params
        # 以分位數做門檻
        thresh = np.percentile(mag, params['mag_percentile'])
        motion_mask = mag >= thresh

        # 保障最小像素比例
        min_pixels = max(1, int(params['min_ratio_pixels'] * h * w))
        if motion_mask.sum() < min_pixels:
            # 沒有足夠動作，不更新（保持舊 ROI）
            return

        ys, xs = np.where(motion_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # 邊界擴張
        expand_x = int(params['expand_ratio'] * (x2 - x1 + 1))
        expand_y = int(params['expand_ratio'] * (y2 - y1 + 1))
        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(w - 1, x2 + expand_x)
        y2 = min(h - 1, y2 + expand_y)

        # 轉為相對座標
        new_roi = (
            float(x1) / float(w),
            float(y1) / float(h),
            float(x2 + 1) / float(w),
            float(y2 + 1) / float(h)
        )

        # 指數移動平均，平滑 ROI
        if self._roi_bbox_norm is None:
            self._roi_bbox_norm = new_roi
        else:
            alpha = params['ema_alpha']
            old = self._roi_bbox_norm
            self._roi_bbox_norm = (
                old[0] * (1 - alpha) + new_roi[0] * alpha,
                old[1] * (1 - alpha) + new_roi[1] * alpha,
                old[2] * (1 - alpha) + new_roi[2] * alpha,
                old[3] * (1 - alpha) + new_roi[3] * alpha,
            )

    def _extract_lucas_kanade_features(self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_idx: int) -> Optional[Dict]:
        """提取Lucas-Kanade光流特徵"""
        # 檢測特徵點
        corners = cv2.goodFeaturesToTrack(
            prev_frame, 
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        if corners is None or len(corners) == 0:
            return self._create_empty_lk_features(frame_idx)
        
        # 計算光流
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, corners, None, **self.lucas_kanade_params
        )
        
        # 選擇好的點，並整理為 (N, 2)
        good_old = corners[status == 1].reshape(-1, 2)
        good_new = next_points[status == 1].reshape(-1, 2)
        
        if len(good_old) == 0:
            return self._create_empty_lk_features(frame_idx)
        
        # 計算運動向量
        motion_vectors = good_new - good_old
        
        # 提取ROI區域特徵
        roi_features = {}
        for roi_name, roi_coords in self.roi_regions.items():
            roi_data = self._extract_roi_point_features(good_old, motion_vectors, roi_coords[0])
            roi_features[roi_name] = roi_data
        
        # 全局統計特徵
        magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
        angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
        
        global_features = {
            'mean_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'max_magnitude': float(np.max(magnitudes)),
            'mean_angle': float(np.mean(angles)),
            'std_angle': float(np.std(angles)),
            'num_points': len(good_old)
        }
        
        return {
            'frame_idx': frame_idx,
            'flow_method': 'lucas_kanade',
            'roi_features': roi_features,
            'global_features': global_features,
            'num_tracked_points': len(good_old)
        }

    def _extract_tvl1_features(self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_idx: int) -> Optional[Dict]:
        """提取TV-L1光流特徵"""
        # 創建TV-L1光流檢測器
        if not hasattr(cv2, 'optflow'):
            raise RuntimeError('opencv-contrib optflow 模組不可用')
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        
        # 計算光流
        flow = tvl1.calc(prev_frame, curr_frame, None)
        
        if flow is None:
            return None
        
        # 計算光流的幅度和角度
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 提取ROI區域特徵
        roi_features = {}
        for roi_name, roi_coords in self.roi_regions.items():
            roi_data = self._extract_roi_flow_features(magnitude, angle, roi_coords[0])
            roi_features[roi_name] = roi_data
        
        # 全局統計特徵
        global_features = {
            'mean_magnitude': float(np.mean(magnitude)),
            'std_magnitude': float(np.std(magnitude)),
            'max_magnitude': float(np.max(magnitude)),
            'mean_angle': float(np.mean(angle)),
            'std_angle': float(np.std(angle))
        }
        
        return {
            'frame_idx': frame_idx,
            'flow_method': 'tvl1',
            'roi_features': roi_features,
            'global_features': global_features,
            'flow_shape': flow.shape[:2]
        }

    def _extract_roi_flow_features(self, magnitude: np.ndarray, angle: np.ndarray, roi_coords: Tuple[float, float, float, float]) -> Dict:
        """提取ROI區域的光流特徵"""
        x1, y1, x2, y2 = roi_coords
        h, w = magnitude.shape
        
        # 轉換為像素座標
        x1_px = int(x1 * w)
        y1_px = int(y1 * h)
        x2_px = int(x2 * w)
        y2_px = int(y2 * h)
        
        # 裁切ROI區域
        roi_mag = magnitude[y1_px:y2_px, x1_px:x2_px]
        roi_ang = angle[y1_px:y2_px, x1_px:x2_px]
        
        if roi_mag.size == 0:
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'histogram_mag': [0.0] * 8,
                'histogram_ang': [0.0] * 8
            }
        
        # 統計特徵
        mean_mag = float(np.mean(roi_mag))
        std_mag = float(np.std(roi_mag))
        max_mag = float(np.max(roi_mag))
        mean_ang = float(np.mean(roi_ang))
        std_ang = float(np.std(roi_ang))
        
        # 直方圖特徵
        hist_mag, _ = np.histogram(roi_mag.flatten(), bins=8, range=(0, np.max(roi_mag) + 1e-6))
        hist_ang, _ = np.histogram(roi_ang.flatten(), bins=8, range=(0, 2*np.pi))
        
        return {
            'mean_magnitude': mean_mag,
            'std_magnitude': std_mag,
            'max_magnitude': max_mag,
            'mean_angle': mean_ang,
            'std_angle': std_ang,
            'histogram_mag': hist_mag.tolist(),
            'histogram_ang': hist_ang.tolist()
        }

    def _extract_roi_point_features(self, points: np.ndarray, motion_vectors: np.ndarray, roi_coords: Tuple[float, float, float, float]) -> Dict:
        """提取ROI區域的點特徵"""
        x1, y1, x2, y2 = roi_coords
        w, h = self.resize_dims
        
        # 轉換為像素座標
        x1_px = x1 * w
        y1_px = y1 * h
        x2_px = x2 * w
        y2_px = y2 * h
        
        # 找出在ROI內的點
        roi_mask = (
            (points[:, 0] >= x1_px) & (points[:, 0] <= x2_px) &
            (points[:, 1] >= y1_px) & (points[:, 1] <= y2_px)
        )
        
        roi_points = points[roi_mask]
        roi_vectors = motion_vectors[roi_mask]
        
        if len(roi_vectors) == 0:
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'num_points': 0
            }
        
        # 計算統計特徵
        magnitudes = np.sqrt(roi_vectors[:, 0]**2 + roi_vectors[:, 1]**2)
        angles = np.arctan2(roi_vectors[:, 1], roi_vectors[:, 0])
        
        return {
            'mean_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'max_magnitude': float(np.max(magnitudes)),
            'mean_angle': float(np.mean(angles)),
            'std_angle': float(np.std(angles)),
            'num_points': len(roi_vectors)
        }

    def _create_empty_lk_features(self, frame_idx: int) -> Dict:
        """創建空的Lucas-Kanade特徵"""
        empty_roi = {
            'mean_magnitude': 0.0,
            'std_magnitude': 0.0,
            'max_magnitude': 0.0,
            'mean_angle': 0.0,
            'std_angle': 0.0,
            'num_points': 0
        }
        
        roi_features = {roi_name: empty_roi.copy() for roi_name in self.roi_regions.keys()}
        
        return {
            'frame_idx': frame_idx,
            'flow_method': 'lucas_kanade',
            'roi_features': roi_features,
            'global_features': empty_roi.copy(),
            'num_tracked_points': 0
        }

    def _apply_temporal_interpolation(self, features: Dict) -> Dict:
        """應用線性插值進行時序對齊"""
        frames_data = features['frames_data']
        if len(frames_data) == 0:
            return features
        
        target_length = self.target_frames
        original_length = len(frames_data)
        
        if original_length == target_length:
            return features
        
        # 準備插值數據
        interpolated_data = []
        
        # 提取所有特徵序列
        feature_sequences = self._extract_feature_sequences(frames_data)
        
        # 執行線性插值
        if original_length >= 2:
            original_indices = np.linspace(0, original_length - 1, original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            
            interpolated_sequences = {}
            for feature_name, sequence in feature_sequences.items():
                if len(sequence) > 0:
                    interpolated_sequences[feature_name] = np.interp(target_indices, original_indices, sequence)
                else:
                    interpolated_sequences[feature_name] = np.zeros(target_length)
        else:
            # 如果只有一個數據點，直接複製
            interpolated_sequences = {}
            for feature_name, sequence in feature_sequences.items():
                if len(sequence) > 0:
                    interpolated_sequences[feature_name] = np.full(target_length, sequence[0])
                else:
                    interpolated_sequences[feature_name] = np.zeros(target_length)
        
        # 重組數據
        for i in range(target_length):
            frame_data = self._reconstruct_frame_data(interpolated_sequences, i, frames_data[0])
            interpolated_data.append(frame_data)
        
        # 更新特徵數據
        features['frames_data'] = interpolated_data
        features['interpolated'] = True
        features['target_frames'] = target_length
        features['original_frames'] = original_length
        
        return features

    def _extract_feature_sequences(self, frames_data: List[Dict]) -> Dict[str, List[float]]:
        """提取所有特徵序列用於插值"""
        sequences = {}
        
        # 全局特徵
        for global_key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
            sequences[f'global_{global_key}'] = [
                frame_data.get('global_features', {}).get(global_key, 0.0) 
                for frame_data in frames_data
            ]
        
        # ROI特徵
        for roi_name in self.roi_regions.keys():
            for feature_key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
                seq_key = f'roi_{roi_name}_{feature_key}'
                sequences[seq_key] = [
                    frame_data.get('roi_features', {}).get(roi_name, {}).get(feature_key, 0.0)
                    for frame_data in frames_data
                ]
            
            # 直方圖特徵
            for hist_type in ['histogram_mag', 'histogram_ang']:
                for bin_idx in range(8):
                    seq_key = f'roi_{roi_name}_{hist_type}_bin_{bin_idx}'
                    sequences[seq_key] = [
                        frame_data.get('roi_features', {}).get(roi_name, {}).get(hist_type, [0.0]*8)[bin_idx]
                        for frame_data in frames_data
                    ]
        
        return sequences

    def _reconstruct_frame_data(self, interpolated_sequences: Dict[str, np.ndarray], frame_idx: int, template_frame: Dict) -> Dict:
        """重構插值後的幀數據"""
        frame_data = {
            'frame_idx': frame_idx,
            'flow_method': template_frame.get('flow_method', self.flow_method),
            'global_features': {},
            'roi_features': {}
        }
        
        # 重構全局特徵
        for global_key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
            seq_key = f'global_{global_key}'
            if seq_key in interpolated_sequences:
                frame_data['global_features'][global_key] = float(interpolated_sequences[seq_key][frame_idx])
        
        # 重構ROI特徵
        for roi_name in self.roi_regions.keys():
            frame_data['roi_features'][roi_name] = {}
            
            # 基本統計特徵
            for feature_key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
                seq_key = f'roi_{roi_name}_{feature_key}'
                if seq_key in interpolated_sequences:
                    frame_data['roi_features'][roi_name][feature_key] = float(interpolated_sequences[seq_key][frame_idx])
            
            # 直方圖特徵
            for hist_type in ['histogram_mag', 'histogram_ang']:
                histogram = []
                for bin_idx in range(8):
                    seq_key = f'roi_{roi_name}_{hist_type}_bin_{bin_idx}'
                    if seq_key in interpolated_sequences:
                        histogram.append(float(interpolated_sequences[seq_key][frame_idx]))
                    else:
                        histogram.append(0.0)
                frame_data['roi_features'][roi_name][hist_type] = histogram
        
        return frame_data

    def extract_normalized_features(self, features: Dict) -> np.ndarray:
        """提取並標準化特徵向量"""
        if not features or not features['frames_data']:
            return np.array([])
        
        frames_data = features['frames_data']
        num_frames = len(frames_data)
        
        # 計算特徵維度
        feature_dim = self._calculate_feature_dimension()
        
        # 初始化特徵矩陣
        feature_matrix = np.zeros((num_frames, feature_dim))
        
        for i, frame_data in enumerate(frames_data):
            feature_vector = self._extract_frame_feature_vector(frame_data)
            feature_matrix[i] = feature_vector
        
        # 標準化處理
        feature_matrix = self._normalize_features(feature_matrix)
        
        return feature_matrix

    def _calculate_feature_dimension(self) -> int:
        """計算特徵維度"""
        # 全局特徵維度: 5個統計特徵
        global_dim = 5
        
        # 每個ROI區域特徵維度: 5個統計特徵 + 2個直方圖(8 bins each) = 21
        roi_dim = 5 + 8 + 8
        num_rois = len(self.roi_regions)
        
        total_dim = global_dim + (roi_dim * num_rois)
        
        return total_dim

    def _extract_frame_feature_vector(self, frame_data: Dict) -> np.ndarray:
        """從幀數據中提取特徵向量"""
        feature_vector = []
        
        # 全局特徵
        global_features = frame_data.get('global_features', {})
        for key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
            feature_vector.append(global_features.get(key, 0.0))
        
        # ROI特徵
        roi_features = frame_data.get('roi_features', {})
        for roi_name in self.roi_regions.keys():
            roi_data = roi_features.get(roi_name, {})
            
            # 統計特徵
            for key in ['mean_magnitude', 'std_magnitude', 'max_magnitude', 'mean_angle', 'std_angle']:
                feature_vector.append(roi_data.get(key, 0.0))
            
            # 直方圖特徵
            for hist_type in ['histogram_mag', 'histogram_ang']:
                histogram = roi_data.get(hist_type, [0.0] * 8)
                feature_vector.extend(histogram)
        
        return np.array(feature_vector)

    def _normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """標準化特徵"""
        if feature_matrix.shape[0] == 0:
            return feature_matrix
        
        # Z-score標準化
        mean = np.mean(feature_matrix, axis=0)
        std = np.std(feature_matrix, axis=0)
        
        # 避免除以零
        std[std == 0] = 1.0
        
        normalized_matrix = (feature_matrix - mean) / std
        
        return normalized_matrix

    def save_features(self, features: Dict, output_path: str):
        """儲存特徵到文件（僅保存.npy檔案）"""
        # 創建輸出目錄
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 只儲存標準化特徵矩陣
        feature_matrix = self.extract_normalized_features(features)
        np.save(output_path, feature_matrix.astype(np.float32))

class KaggleOpticalFlowExtractor:
    def __init__(self, 
                 dataset_root: str,
                 output_root: str = None,
                 num_workers: int = None,
                 flow_method: str = 'farneback',
                 confidence_threshold: float = 0.5):
        """
        Kaggle Tesla P100 光流特徵提取器
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root) if output_root else self.dataset_root.parent / "optical_flow_features"
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.flow_method = flow_method
        self.confidence_threshold = confidence_threshold
            
        # 確保輸出目錄存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        print("🚀 KaggleOpticalFlowExtractor 初始化完成（已淘汰，請改用本地 MPS 流程或 MpsOpticalFlowExtractor）")
        print(f"📂 資料集路徑: {self.dataset_root}")
        print(f"📁 輸出路徑: {self.output_root}")
        print(f"⚡ 工作進程數: {self.num_workers}")
        print(f"🌊 光流方法: {flow_method}")
        print(f"🎯 GPU: Tesla P100 優化模式")

    def discover_videos(self) -> dict:
        """發現所有影片文件"""
        videos_by_class = {}
        # 擴充支援更多影片格式，包含大小寫
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV', 
                           '.m4v', '.M4V', '.webm', '.WEBM', '.flv', '.FLV'}
        
        print(f"🔍 掃描光流資料集目錄: {self.dataset_root}")
        
        if not self.dataset_root.exists():
            print(f"❌ 資料集目錄不存在: {self.dataset_root}")
            return videos_by_class
            
        subdirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        print(f"📁 找到 {len(subdirs)} 個子目錄")
        
        for class_dir in subdirs:
            if class_dir.is_dir():
                class_name = class_dir.name
                videos = []
                
                # 先檢查目錄是否可讀取
                try:
                    all_files = list(class_dir.iterdir())
                    print(f"📂 {class_name}: 總共 {len(all_files)} 個檔案")
                    
                    for video_file in all_files:
                        if video_file.is_file() and video_file.suffix in video_extensions:
                            videos.append(str(video_file))
                        elif video_file.is_file():
                            # 調試：顯示非影片檔案的副檔名
                            if len(videos) < 3:  # 只顯示前幾個
                                print(f"   🔍 發現其他檔案: {video_file.name} (副檔名: '{video_file.suffix}')")
                    
                    if videos:
                        videos_by_class[class_name] = sorted(videos)
                        print(f"   ✅ {class_name}: 找到 {len(videos)} 個影片")
                    else:
                        print(f"   ❌ {class_name}: 未找到影片檔案")
                        
                except Exception as e:
                    print(f"   ❌ 無法讀取目錄 {class_name}: {e}")
                    
        print(f"📊 總結: {len(videos_by_class)} 個類別包含影片")
        return videos_by_class

    def _read_failed_video_paths_from_report(self, report_path: Optional[str] = None) -> List[str]:
        """從處理報告中讀取失敗/錯誤影片的完整路徑清單"""
        report = Path(report_path) if report_path else (self.output_root / "optical_flow_processing_report.json")
        if not report.exists():
            print(f"❌ 找不到處理報告: {report}")
            return []
        try:
            with open(report, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 無法讀取報告 {report}: {e}")
            return []

        failed_paths: List[str] = []
        for item in data.get('details', []):
            if item.get('status') in ['failed', 'error'] and item.get('video_path'):
                failed_paths.append(item['video_path'])

        # 去重 & 僅保留仍存在於資料集的影片
        unique_paths = []
        seen = set()
        for p in failed_paths:
            if p not in seen:
                seen.add(p)
                if Path(p).exists():
                    unique_paths.append(p)
                else:
                    # 嘗試以資料集根目錄重新定位（保險）
                    candidate = self.dataset_root / Path(p).name[: Path(p).name.find('_')] / Path(p).name
                    if candidate.exists():
                        unique_paths.append(str(candidate))
        print(f"🧾 從報告取得需重試影片: {len(unique_paths)} 筆")
        return unique_paths

    def _resolve_video_names_to_paths(self, names: List[str]) -> List[str]:
        """將檔名（如 apple_0850.mp4）解析為資料集內的完整路徑"""
        resolved: List[str] = []
        for name in names:
            base = Path(name).name
            # 先用類別=底線前綴的規則
            class_name = base.split('_')[0] if '_' in base else None
            candidates: List[Path] = []
            if class_name:
                candidates.append(self.dataset_root / class_name / base)
            # 後備方案：全域搜尋（僅限檔名匹配）
            if not any(p.exists() for p in candidates):
                for p in self.dataset_root.rglob(base):
                    if p.is_file():
                        candidates.append(p)
                        break
            # 收斂為第一個存在的候選
            target = next((str(p) for p in candidates if p.exists()), None)
            if target:
                resolved.append(target)
            else:
                print(f"⚠️ 找不到影片: {name}")
        print(f"🧭 指定檔名解析完成，可處理: {len(resolved)} / {len(names)}")
        return resolved

    def _process_specific_video_paths(self, video_paths: List[str]) -> dict:
        """並行處理指定的一組影片路徑"""
        if not video_paths:
            print("❌ 沒有可處理的影片清單")
            return {}

        batches = self.create_batches(video_paths)
        print(f"📦 重試批次數: {len(batches)} ；影片數: {len(video_paths)}")

        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'error': 0,
            'details': [],
            'start_time': time.time()
        }

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {executor.submit(self.process_video_batch, batch): batch for batch in batches}
            with tqdm(total=len(video_paths), desc="重試影片", unit="影片") as pbar:
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    for result in batch_results:
                        results['details'].append(result)
                        results[result['status']] += 1
                        pbar.update(1)
                    pbar.set_postfix({
                        '成功': results['success'],
                        '失敗': results['failed'],
                        '跳過': results['skipped'],
                        '錯誤': results['error']
                    })

        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']

        # 輸出重試報告
        retry_report = self.output_root / "optical_flow_retry_report.json"
        try:
            with open(retry_report, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ 無法寫入重試報告: {e}")

        return results

    def retry_failed_from_report(self, report_path: Optional[str] = None) -> dict:
        """只針對報告中的失敗/錯誤影片做再次提取"""
        failed_paths = self._read_failed_video_paths_from_report(report_path)
        if not failed_paths:
            print("✅ 沒有可重試的失敗影片（或找不到報告）")
            return {}
        print("🚀 開始重試報告中的失敗影片...")
        return self._process_specific_video_paths(failed_paths)

    def retry_specific_videos(self, names: List[str]) -> dict:
        """只針對手動指定的影片檔名做再次提取（如 ['apple_0850.mp4']）"""
        video_paths = self._resolve_video_names_to_paths(names)
        if not video_paths:
            print("❌ 指定的影片皆無法解析路徑，無法重試")
            return {}
        print("🚀 開始重試指定清單影片...")
        return self._process_specific_video_paths(video_paths)

    def get_processing_stats(self, videos_by_class: dict) -> dict:
        """獲取處理統計信息"""
        stats = {
            'total_classes': len(videos_by_class),
            'total_videos': sum(len(videos) for videos in videos_by_class.values()),
            'videos_per_class': {},
            'estimated_time': 0
        }
        
        for class_name, videos in videos_by_class.items():
            stats['videos_per_class'][class_name] = len(videos)
        
        # Tesla P100 光流計算優化後估算時間 (每個影片約4-6秒)
        avg_time_per_video = 5.0
        stats['estimated_time'] = stats['total_videos'] * avg_time_per_video / self.num_workers
        
        return stats

    def process_video_batch(self, video_paths_batch: list) -> list:
        """處理一批影片的工作函數"""
        extractor = OpticalFlowExtractor(
            target_frames=100,
            flow_method=self.flow_method,
            confidence_threshold=self.confidence_threshold
        )
        results = []
        
        for video_path in video_paths_batch:
            try:
                # 計算輸出路徑，保持和資料集相同的目錄結構
                video_path_obj = Path(video_path)
                class_name = video_path_obj.parent.name
                output_dir = self.output_root / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 確保輸出目錄存在
                if not output_dir.exists():
                    print(f"⚠️ 創建光流輸出目錄: {output_dir}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = output_dir / f"{video_path_obj.stem}.npy"
                
                # 跳過已存在的文件
                if output_path.exists():
                    results.append({
                        'video_path': video_path,
                        'output_path': str(output_path),
                        'status': 'skipped',
                        'message': '文件已存在'
                    })
                    continue
                
                # 提取特徵
                features = extractor.extract_video_features(video_path)
                
                if features is not None:
                    # 儲存特徵
                    extractor.save_features(features, str(output_path))
                    
                    # 獲取特徵統計
                    feature_matrix = extractor.extract_normalized_features(features)
                    
                    results.append({
                        'video_path': video_path,
                        'output_path': str(output_path),
                        'status': 'success',
                        'original_frames': features.get('original_frames', len(features['frames_data'])),
                        'target_frames': features.get('target_frames', 100),
                        'feature_shape': feature_matrix.shape,
                        'duration': features['video_info']['duration'],
                        'flow_method': features['video_info']['flow_method']
                    })
                else:
                    results.append({
                        'video_path': video_path,
                        'output_path': str(output_path),
                        'status': 'failed',
                        'message': '光流特徵提取失敗'
                    })
                    
            except Exception as e:
                results.append({
                    'video_path': video_path,
                    'output_path': '',
                    'status': 'error',
                    'message': str(e)
                })
        
        return results

    def create_batches(self, all_videos: list, batch_size: int = None) -> list:
        """將影片列表分成批次"""
        if batch_size is None:
            batch_size = max(1, len(all_videos) // (self.num_workers * 4))
        
        batches = []
        for i in range(0, len(all_videos), batch_size):
            batches.append(all_videos[i:i + batch_size])
        
        return batches

    def process_dataset(self, limit_per_class: int = None) -> dict:
        """處理整個資料集"""
        print(f"🔍 掃描影片文件...")
        videos_by_class = self.discover_videos()
        
        if not videos_by_class:
            print("❌ 未找到任何影片文件")
            return {}
        
        # 限制每個類別的影片數量
        if limit_per_class:
            for class_name in videos_by_class:
                videos_by_class[class_name] = videos_by_class[class_name][:limit_per_class]
        
        # 獲取統計信息
        stats = self.get_processing_stats(videos_by_class)
        
        print(f"📊 發現 {stats['total_classes']} 個類別，共 {stats['total_videos']} 個影片")
        print(f"⏱️  估算處理時間: {stats['estimated_time']/60:.1f} 分鐘")
        
        # Kaggle自動模式
        print("🤖 Kaggle模式，自動開始處理...")
        
        # 創建所有影片路徑列表
        all_videos = []
        for class_name, videos in videos_by_class.items():
            all_videos.extend(videos)
        
        # 創建批次
        batches = self.create_batches(all_videos)
        print(f"📦 創建 {len(batches)} 個批次")
        
        # 處理統計
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'error': 0,
            'details': [],
            'start_time': time.time()
        }
        
        # 使用進程池並行處理
        print(f"🚀 開始並行處理 ({self.num_workers} 個工作進程)...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任務
            future_to_batch = {
                executor.submit(self.process_video_batch, batch): batch 
                for batch in batches
            }
            
            # 使用進度條顯示處理進度
            with tqdm(total=len(all_videos), desc="處理影片", unit="影片") as pbar:
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    
                    for result in batch_results:
                        results['details'].append(result)
                        results[result['status']] += 1
                        pbar.update(1)
                        
                        # 更新進度條描述
                        pbar.set_postfix({
                            '成功': results['success'],
                            '失敗': results['failed'],
                            '跳過': results['skipped'],
                            '錯誤': results['error']
                        })
        
        # 計算處理時間
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        # 儲存處理報告
        report_path = self.output_root / "optical_flow_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return results

    def validate_extracted_features(self, sample_size: int = 100) -> dict:
        """驗證提取的特徵品質"""
        features_path = self.output_root
        
        if not features_path.exists():
            print(f"❌ 特徵目錄不存在: {features_path}")
            return {}
        
        print(f"🔍 驗證光流特徵目錄: {features_path}")
        
        # 收集所有.npy文件
        all_npy_files = []
        for class_dir in features_path.iterdir():
            if class_dir.is_dir():
                npy_files = list(class_dir.glob('*.npy'))
                for npy_file in npy_files:
                    all_npy_files.append((class_dir.name, npy_file))
        
        if not all_npy_files:
            print("❌ 未找到任何.npy特徵文件")
            return {}
        
        print(f"📊 找到 {len(all_npy_files)} 個光流特徵文件")
        
        # 隨機抽樣
        sample_files = random.sample(all_npy_files, min(sample_size, len(all_npy_files)))
        
        validation_stats = {
            'total_files': len(all_npy_files),
            'sample_size': len(sample_files),
            'shapes': [],
            'feature_stats': {
                'min_vals': [],
                'max_vals': [],
                'mean_vals': [],
                'std_vals': []
            },
            'classes': {},
            'corrupted_files': [],
            'dimension_consistency': True,
            'flow_method': self.flow_method
        }
        
        expected_shape = None
        
        print(f"🧪 驗證 {len(sample_files)} 個樣本...")
        
        for class_name, npy_file in tqdm(sample_files, desc="驗證光流特徵"):
            try:
                # 載入特徵
                features = np.load(npy_file)
                
                # 檢查形狀
                if expected_shape is None:
                    expected_shape = features.shape
                    print(f"📏 預期光流特徵形狀: {expected_shape}")
                
                validation_stats['shapes'].append(features.shape)
                
                if features.shape != expected_shape:
                    validation_stats['dimension_consistency'] = False
                    print(f"⚠️  形狀不一致: {npy_file.name} -> {features.shape}")
                
                # 統計資訊
                validation_stats['feature_stats']['min_vals'].append(np.min(features))
                validation_stats['feature_stats']['max_vals'].append(np.max(features))
                validation_stats['feature_stats']['mean_vals'].append(np.mean(features))
                validation_stats['feature_stats']['std_vals'].append(np.std(features))
                
                # 類別統計
                if class_name not in validation_stats['classes']:
                    validation_stats['classes'][class_name] = 0
                validation_stats['classes'][class_name] += 1
                
                # 檢查是否有異常值
                if np.isnan(features).any() or np.isinf(features).any():
                    validation_stats['corrupted_files'].append(str(npy_file))
                    print(f"❌ 發現異常值: {npy_file.name}")
                
            except Exception as e:
                validation_stats['corrupted_files'].append(str(npy_file))
                print(f"❌ 載入失敗: {npy_file.name} -> {e}")
        
        # 統計分析
        print(f"\n📈 光流特徵驗證結果:")
        print(f"✅ 成功載入: {len(sample_files) - len(validation_stats['corrupted_files'])} 個文件")
        print(f"❌ 損壞文件: {len(validation_stats['corrupted_files'])} 個")
        print(f"📏 形狀一致性: {'✅ 通過' if validation_stats['dimension_consistency'] else '❌ 失敗'}")
        print(f"🌊 光流方法: {self.flow_method}")
        
        if expected_shape:
            print(f"📊 特徵維度: {expected_shape[0]} 幀 × {expected_shape[1]} 光流特徵")
        
        # 特徵統計
        if validation_stats['feature_stats']['min_vals']:
            min_vals = validation_stats['feature_stats']['min_vals']
            max_vals = validation_stats['feature_stats']['max_vals']
            mean_vals = validation_stats['feature_stats']['mean_vals']
            std_vals = validation_stats['feature_stats']['std_vals']
            
            print(f"📊 光流特徵數值統計:")
            print(f"   最小值範圍: {np.min(min_vals):.3f} 至 {np.max(min_vals):.3f}")
            print(f"   最大值範圍: {np.min(max_vals):.3f} 至 {np.max(max_vals):.3f}")
            print(f"   平均值範圍: {np.min(mean_vals):.3f} 至 {np.max(mean_vals):.3f}")
            print(f"   標準差範圍: {np.min(std_vals):.3f} 至 {np.max(std_vals):.3f}")
        
        # 類別分佈
        print(f"\n📊 類別分佈（樣本）:")
        for class_name, count in sorted(validation_stats['classes'].items()):
            print(f"   {class_name}: {count} 個樣本")
        
        # 生成驗證報告
        validation_stats['summary'] = {
            'total_files': len(all_npy_files),
            'sample_validated': len(sample_files),
            'success_rate': (len(sample_files) - len(validation_stats['corrupted_files'])) / len(sample_files),
            'dimension_consistent': validation_stats['dimension_consistency'],
            'expected_shape': expected_shape,
            'flow_method': self.flow_method
        }
        
        # 儲存報告
        report_path = features_path / "optical_flow_validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📈 光流驗證報告已儲存至: {report_path}")
        
        return validation_stats

    def create_features_zip(self, zip_name: str = None) -> str:
        """打包特徵文件為zip"""
        if zip_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            zip_name = f"optical_flow_features_{self.flow_method}_{timestamp}.zip"
        
        zip_path = self.output_root.parent / zip_name
        
        print(f"📦 開始打包光流特徵文件...")
        
        # 收集所有.npy文件
        all_npy_files = []
        for class_dir in self.output_root.iterdir():
            if class_dir.is_dir():
                npy_files = list(class_dir.glob('*.npy'))
                all_npy_files.extend(npy_files)
        
        if not all_npy_files:
            print("❌ 未找到任何.npy特徵文件")
            return ""
        
        print(f"📊 找到 {len(all_npy_files)} 個光流特徵文件")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for npy_file in tqdm(all_npy_files, desc="打包文件"):
                # 保持目錄結構
                arcname = npy_file.relative_to(self.output_root)
                zipf.write(npy_file, arcname)
            
            # 添加報告文件
            report_files = ['optical_flow_processing_report.json', 'optical_flow_validation_report.json']
            for report_file in report_files:
                report_path = self.output_root / report_file
                if report_path.exists():
                    zipf.write(report_path, report_file)
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"✅ 光流特徵打包完成: {zip_path}")
        print(f"📊 壓縮檔大小: {zip_size_mb:.1f} MB")
        print(f"🌊 光流方法: {self.flow_method}")
        
        return str(zip_path)

    def print_results_summary(self, results: dict):
        """打印結果摘要"""
        if not results:
            return
            
        print(f"\n📈 光流特徵處理完成摘要:")
        print(f"⏱️  總耗時: {results['total_time']/60:.1f} 分鐘")
        print(f"✅ 成功: {results['success']} 個影片")
        print(f"⏭️  跳過: {results['skipped']} 個影片")
        print(f"❌ 失敗: {results['failed']} 個影片")
        print(f"💥 錯誤: {results['error']} 個影片")
        print(f"🌊 光流方法: {self.flow_method}")
        
        total_processed = results['success'] + results['failed'] + results['skipped'] + results['error']
        if total_processed > 0:
            success_rate = (results['success'] / total_processed) * 100
            print(f"📊 成功率: {success_rate:.1f}%")
        
        if results['failed'] > 0 or results['error'] > 0:
            print(f"\n⚠️  失敗的影片:")
            for detail in results['details']:
                if detail['status'] in ['failed', 'error']:
                    print(f"   - {Path(detail['video_path']).name}: {detail.get('message', '未知錯誤')}")

    def run_complete_pipeline(self, limit_per_class: int = None, sample_size: int = 100):
        """執行完整的光流特徵提取、驗證和打包流程"""
        print("🚀 開始完整的光流特徵提取流程...")
        print(f"🌊 使用光流方法: {self.flow_method}")
        
        # 1. 光流特徵提取
        print("\n📝 步驟1: 光流特徵提取")
        results = self.process_dataset(limit_per_class=limit_per_class)
        self.print_results_summary(results)
        
        # 2. 特徵驗證
        print("\n📝 步驟2: 光流特徵驗證")
        validation_stats = self.validate_extracted_features(sample_size=sample_size)
        
        # 3. 打包特徵
        print("\n📝 步驟3: 打包光流特徵")
        zip_path = self.create_features_zip()
        print("🎉 完整光流處理流程完成!")
        print(f"📦 光流特徵壓縮檔: {zip_path}")
        print(f"🌊 光流方法: {self.flow_method}")
        return {
            'extraction_results': results,
            'validation_stats': validation_stats,
            'zip_path': zip_path,
            'flow_method': self.flow_method
        }


# --- MPS 友善包裝：穩定輸出與可重現性，預設輸出到專案 features/optical_flow_features ---
class MpsOpticalFlowExtractor:
    def __init__(self,
                 target_frames: int = 100,
                 resize_dims: Tuple[int, int] = (224, 224),
                 flow_method: str = 'farneback',
                 output_root: Optional[str] = None,
                 roi_mode: str = 'static'):
        self.target_frames = int(target_frames)
        self.resize_dims = resize_dims
        self.flow_method = flow_method
        self.roi_mode = roi_mode
        self.output_root = Path(output_root) if output_root else OUTPUT_ROOT

    def extract_and_save(self, video_path: str, class_name: Optional[str] = None) -> Path:
        """抽取單支影片光流特徵，保存為 .npy 並回傳路徑。"""
        extractor = OpticalFlowExtractor(
            target_frames=self.target_frames,
            flow_method=self.flow_method,
            resize_dims=self.resize_dims,
            roi_mode=self.roi_mode,
        )
        vp = Path(video_path)
        cls = class_name or vp.parent.name
        out_dir = self.output_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (vp.stem + ".npy")

        # 跳過已存在
        if out_path.exists():
            return out_path

        feats = extractor.extract_video_features(str(vp))
        if feats is None:
            # 寫入零矩陣以保持配對，形狀 [T, 8]
            np.save(out_path.as_posix(), np.zeros((self.target_frames, 8), dtype=np.float32))
            return out_path

        mat = extractor.extract_normalized_features(feats)
        # 線性插值對齊長度（保險）
        if mat.shape[0] != self.target_frames:
            x_old = np.linspace(0, 1, mat.shape[0], dtype=np.float32)
            x_new = np.linspace(0, 1, self.target_frames, dtype=np.float32)
            aligned = np.vstack([np.interp(x_new, x_old, mat[:, d]) for d in range(mat.shape[1])]).T
            mat = aligned.astype(np.float32)

        np.save(out_path.as_posix(), mat.astype(np.float32))
        return out_path


# ---- 本地 MPS 資料集批次處理（頂層函式，便於 main() 直接呼叫） ----
_PROC_OF_EXTRACTOR = None


def _discover_videos_local(dataset_root: Path) -> Dict[str, List[str]]:
    videos_by_class: Dict[str, List[str]] = {}
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".flv",
            ".MP4", ".AVI", ".MOV", ".MKV", ".M4V", ".WEBM", ".FLV"}
    if not dataset_root.exists():
        return videos_by_class
    for class_dir in sorted([d for d in dataset_root.iterdir() if d.is_dir()]):
        files = [f.as_posix() for f in class_dir.iterdir() if f.is_file() and f.suffix in exts]
        if files:
            videos_by_class[class_dir.name] = sorted(files)
    return videos_by_class


def _init_worker_of(target_frames: int, resize_dims: Tuple[int, int], flow_method: str, roi_mode: str):
    global _PROC_OF_EXTRACTOR
    _PROC_OF_EXTRACTOR = MpsOpticalFlowExtractor(
        target_frames=target_frames,
        resize_dims=resize_dims,
        flow_method=flow_method,
        roi_mode=roi_mode,
    )


def _worker_process_video_of(task: Tuple[str, str]) -> dict:
    cls, vp = task
    try:
        global _PROC_OF_EXTRACTOR
        if _PROC_OF_EXTRACTOR is None:
            _PROC_OF_EXTRACTOR = MpsOpticalFlowExtractor()
        out_path = _PROC_OF_EXTRACTOR.extract_and_save(vp, class_name=cls)
        arr = np.load(out_path)
        return {"video": vp, "status": "success", "shape": list(arr.shape), "output": out_path.as_posix()}
    except Exception as e:
        return {"video": vp, "status": "error", "msg": str(e)}


def run_dataset_extraction_local(dataset_root: Optional[str] = None,
                                 target_frames: int = 100,
                                 resize_dims: Tuple[int, int] = (224, 224),
                                 flow_method: str = 'farneback',
                                 roi_mode: str = 'static',
                                 limit_per_class: Optional[int] = None,
                                 parallel: bool = True,
                                 num_workers: int = NUM_WORKERS) -> dict:
    ds_root = Path(dataset_root) if dataset_root else DATASET_PATH
    print("🚀 本地 MPS 光流特徵提取")
    print(f"📂 資料集: {ds_root}")
    print(f"📁 輸出:   {OUTPUT_ROOT}")

    vids = _discover_videos_local(ds_root)
    if not vids:
        print("❌ 找不到任何影片，請確認資料集路徑")
        return {}

    if limit_per_class:
        vids = {k: v[:limit_per_class] for k, v in vids.items()}

    total = sum(len(v) for v in vids.values())
    print(f"📊 類別: {len(vids)}，影片總數: {total}")

    results = {"success": 0, "failed": 0, "details": [], "start_time": time.time()}

    all_videos: List[Tuple[str, str]] = []
    for cls, paths in vids.items():
        for vp in paths:
            all_videos.append((cls, vp))

    if not parallel:
        extractor = MpsOpticalFlowExtractor(target_frames=target_frames, resize_dims=resize_dims, flow_method=flow_method, roi_mode=roi_mode)
        with tqdm(total=total, desc="處理影片", unit="支", dynamic_ncols=True) as pbar:
            for cls, vp in all_videos:
                try:
                    out_path = extractor.extract_and_save(vp, class_name=cls)
                    arr = np.load(out_path)
                    results["success"] += 1
                    results["details"].append({"video": vp, "status": "success", "shape": list(arr.shape)})
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append({"video": vp, "status": "error", "msg": str(e)})
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"成功": results["success"], "失敗": results["failed"], "最近": Path(vp).name})
    else:
        print(f"🚀 平行處理開啟（workers={num_workers}）...")
        with ProcessPoolExecutor(max_workers=num_workers,
                                 initializer=_init_worker_of,
                                 initargs=(target_frames, resize_dims, flow_method, roi_mode)) as ex:
            chunk_size = max(1, len(all_videos)//(num_workers*4) or 1)
            with tqdm(total=total, desc="處理影片", unit="支", dynamic_ncols=True) as pbar:
                for res in ex.map(_worker_process_video_of, all_videos, chunksize=chunk_size):
                    results["details"].append(res)
                    status = res.get("status", "error")
                    if status == "success":
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                    pbar.update(1)
                    name = Path(res.get("video", "")).name if res else ""
                    pbar.set_postfix({"成功": results["success"], "失敗": results["failed"], "最近": name})

    results["end_time"] = time.time()
    results["total_time"] = results["end_time"] - results["start_time"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with open((OUTPUT_ROOT / "optical_flow_processing_report_local.json").as_posix(), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print("\n📈 完成。")
    print(f"⏱️  耗時: {results['total_time']/60:.1f} 分鐘")
    print(f"✅ 成功: {results['success']} | ❌ 失敗: {results['failed']}")
    return results


def main():
    # 本地預設：與訓練程式讀取路徑一致
    run_dataset_extraction_local(
        dataset_root=str(DATASET_PATH),
        target_frames=100,
        resize_dims=(224, 224),
        flow_method='farneback',
        roi_mode='static',
        limit_per_class=None,
        parallel=True,
    )


if __name__ == "__main__":
    main()