import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    def __init__(self):
        # 移除在初始化時創建segmentation物件
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
    def iter_segmented_frames(self, input_path, bg_color=(0, 0, 0, 0)):
        """
        逐幀回傳已去背的畫面（BGR），不保存影片。
        使用 MediaPipe SelfieSegmentation，背景以 bg_color 混合。
        """
        try:
            with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    return

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for _ in range(frame_count if frame_count > 0 else 10**9):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = segmentation.process(frame_rgb)
                    mask = getattr(results, 'segmentation_mask', None)

                    # 若無遮罩，直接回傳原始幀
                    if mask is None:
                        yield frame
                        continue

                    mask = cv2.resize(mask, (width, height))
                    mask4 = np.stack((mask,) * 4, axis=-1)  # 4-channel mask

                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    bg_image = np.ones((height, width, 4), dtype=np.uint8)
                    bg_image[:] = bg_color

                    output_rgba = frame_rgba * mask4 + bg_image * (1 - mask4)
                    output_bgr = cv2.cvtColor(output_rgba.astype(np.uint8), cv2.COLOR_BGRA2BGR)
                    yield output_bgr

                cap.release()
        except Exception:
            # 靜默失敗，呼叫端可自行 fallback
            return

    def process_video(self, input_path, output_path, bg_color=(0, 0, 0, 0)):
        """處理單一影片的去背"""
        try:
            # 確保輸出目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 為每個影片創建獨立的segmentation物件
            with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    print(f"無法開啟影片: {input_path}")
                    return False
                
                # 獲取影片屬性
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 設定輸出影片編碼器與格式
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編碼
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # 處理每一幀
                for _ in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 轉換為RGB (MediaPipe需要RGB格式)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 進行分割，使用獨立的segmentation物件
                    results = segmentation.process(frame_rgb)
                    mask = results.segmentation_mask
                    
                    # 生成RGBA圖像
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    
                    # 確保遮罩和幀有相同的尺寸
                    if mask is not None:
                        mask = cv2.resize(mask, (width, height))
                        mask = np.stack((mask,) * 4, axis=-1)  # 將遮罩擴展為4通道
                        
                        # 創建透明背景圖像
                        bg_image = np.ones((height, width, 4), dtype=np.uint8)
                        bg_image[:] = bg_color
                        
                        # 混合前景和背景
                        output_frame = np.zeros((height, width, 4), dtype=np.uint8)
                        output_frame = frame_rgba * mask + bg_image * (1 - mask)
                        
                        # 轉回BGR格式以保存
                        output_frame_bgr = cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_BGRA2BGR)
                        out.write(output_frame_bgr)
                        
                cap.release()
                out.release()
                return True
            
        except Exception as e:
            print(f"處理影片時出錯: {input_path}, 錯誤: {str(e)}")
            return False

    def segment_frames(self, frames, bg_color=(0, 0, 0, 0)):
        """對一串 BGR 幀做去背，逐幀回傳 BGR，完全不寫檔。"""
        try:
            with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
                # 嘗試從第一幀推測尺寸
                width = None
                height = None
                for frame in frames:
                    if frame is None:
                        continue
                    if width is None or height is None:
                        height, width = frame.shape[:2]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = segmentation.process(frame_rgb)
                    mask = getattr(results, 'segmentation_mask', None)
                    if mask is None:
                        yield frame
                        continue
                    mask = cv2.resize(mask, (width, height))
                    mask4 = np.stack((mask,) * 4, axis=-1)
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    bg_image = np.ones((height, width, 4), dtype=np.uint8)
                    bg_image[:] = bg_color
                    output_rgba = frame_rgba * mask4 + bg_image * (1 - mask4)
                    output_bgr = cv2.cvtColor(output_rgba.astype(np.uint8), cv2.COLOR_BGRA2BGR)
                    yield output_bgr
        except Exception:
            # 靜默失敗，呼叫端可自行 fallback
            for frame in frames:
                if frame is not None:
                    yield frame

    def process_directory(self, input_dir, output_dir, max_workers=4):
        """處理整個目錄下的所有影片"""
        if not os.path.exists(input_dir):
            print(f"輸入目錄不存在: {input_dir}")
            return
            
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取所有影片文件
        video_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
                    rel_dir = os.path.relpath(root, input_dir)
                    src_path = os.path.join(root, file)
                    
                    # 保持相同的目錄結構
                    if rel_dir == '.':
                        dest_dir = output_dir
                    else:
                        dest_dir = os.path.join(output_dir, rel_dir)
                    
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, f"{file}")
                    
                    video_files.append((src_path, dest_path))
        
        print(f"發現 {len(video_files)} 個影片檔案待處理")
        
        # 使用多線程處理影片
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(
                    lambda x: self.process_video(x[0], x[1]), 
                    video_files
                ), 
                total=len(video_files), 
                desc="處理影片"
            ))
        
        success_count = sum(results)
        print(f"成功處理 {success_count}/{len(video_files)} 個影片")

if __name__ == "__main__":
    # 獲取目前目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 影片輸入輸出目錄
    input_dir = os.path.join(current_dir, "bai_dataset")
    output_dir = os.path.join(current_dir, "videos_segmented")
    
    # 初始化處理器並處理影片
    processor = VideoProcessor()
    processor.process_directory(input_dir, output_dir) 