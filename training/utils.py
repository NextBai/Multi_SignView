"""
Training Utilities
訓練工具模組
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class EarlyStopping:
    """早停機制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.is_better = self._get_is_better_func()

    def _get_is_better_func(self):
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            return lambda current, best: current > best + self.min_delta

    def __call__(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self):
        self.counter = 0
        self.best_value = None


class ModelCheckpoint:
    """模型檢查點管理"""

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_acc',
        mode: str = 'max',
        save_best_only: bool = True,
        filename_format: str = 'model_epoch_{epoch:03d}_acc_{acc:.3f}.pth'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename_format = filename_format
        self.best_value = None
        self.is_better = self._get_is_better_func()

    def _get_is_better_func(self):
        if self.mode == 'min':
            return lambda current, best: current < best
        else:  # mode == 'max'
            return lambda current, best: current > best

    def update(self, current_value: float, model: nn.Module, epoch: int) -> bool:
        """更新檢查點，返回是否為最佳模型"""
        is_best = False

        if self.best_value is None or self.is_better(current_value, self.best_value):
            self.best_value = current_value
            is_best = True

        if not self.save_best_only or is_best:
            filename = self.filename_format.format(epoch=epoch, acc=current_value)
            filepath = self.save_dir / filename
            torch.save(model.state_dict(), filepath)

            if is_best:
                # 同時保存為best模型
                best_filepath = self.save_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_filepath)

        return is_best


class TrainingLogger:
    """訓練日誌記錄器"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.history = []

    def log(self, epoch: int, metrics: Dict[str, float], extra_info: str = ""):
        """記錄訓練日誌"""
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            'metrics': metrics,
            'extra_info': extra_info
        }
        self.history.append(log_entry)

        # 控制台輸出
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch:3d}: {metrics_str} {extra_info}")

        # 文件輸出
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch:3d}: {metrics_str} {extra_info}\n")

    def save_history(self, filepath: str):
        """保存訓練歷史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class MetricTracker:
    """指標追蹤器"""

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, **kwargs):
        """更新指標"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []

            self.metrics[key].append(value)

    def get_average(self, key: str) -> float:
        """獲取指標平均值"""
        if key in self.metrics and self.metrics[key]:
            return sum(self.metrics[key]) / len(self.metrics[key])
        return 0.0

    def reset(self):
        """重置當前epoch的指標"""
        for key in self.metrics:
            self.history[key].extend(self.metrics[key])
            self.metrics[key] = []

    def get_history(self, key: str) -> List[float]:
        """獲取指標歷史"""
        return self.history.get(key, [])


class LearningRateScheduler:
    """學習率調度器"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'cosine',
        **kwargs
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type

        if schedule_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif schedule_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.5)
            )
        elif schedule_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5)
            )
        else:
            raise ValueError(f"不支援的調度類型: {schedule_type}")

    def step(self, metric: Optional[float] = None):
        """調度學習率"""
        if self.schedule_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']


class ProgressBar:
    """進度條"""

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, step: int = 1):
        """更新進度"""
        self.current += step
        self._display()

    def _display(self):
        """顯示進度條"""
        percent = self.current / self.total
        bar_length = 50
        filled_length = int(bar_length * percent)

        bar = '█' * filled_length + '▒' * (bar_length - filled_length)
        elapsed_time = time.time() - self.start_time
        eta = elapsed_time / percent - elapsed_time if percent > 0 else 0

        print(f'\r{self.desc} |{bar}| {self.current}/{self.total} '
              f'({percent:.1%}) ETA: {eta:.1f}s', end='', flush=True)

        if self.current >= self.total:
            print()  # 換行


class GradientClipping:
    """梯度裁剪"""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model: nn.Module) -> float:
        """執行梯度裁剪"""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        ).item()


class ModelSummary:
    """模型摘要工具"""

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """計算模型參數"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }

    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """獲取模型大小（MB）"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    @staticmethod
    def print_summary(model: nn.Module, model_name: str = "Model"):
        """打印模型摘要"""
        params = ModelSummary.count_parameters(model)
        size_mb = ModelSummary.get_model_size(model)

        print(f"\n📊 {model_name} 摘要:")
        print(f"   總參數數量: {params['total']:,}")
        print(f"   可訓練參數: {params['trainable']:,}")
        print(f"   凍結參數: {params['frozen']:,}")
        print(f"   模型大小: {size_mb:.2f} MB")


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config = {}
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """載入配置文件"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"載入配置文件失敗: {e}")

    def save_config(self, config_file: str):
        """保存配置文件"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default: Any = None):
        """獲取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """設置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


def set_random_seed(seed: int = 42):
    """設置隨機種子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # 確保可重現性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """獲取計算設備"""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 使用 CPU")

    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
):
    """保存完整檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time(),
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """載入完整檢查點"""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


if __name__ == "__main__":
    # 測試工具函數
    print("🧪 測試訓練工具...")

    # 測試早停
    early_stopping = EarlyStopping(patience=3)
    test_losses = [1.0, 0.8, 0.9, 0.85, 0.87, 0.86]
    for i, loss in enumerate(test_losses):
        if early_stopping(loss):
            print(f"Early stopping triggered at step {i}")
            break

    # 測試指標追蹤
    tracker = MetricTracker()
    tracker.update(loss=1.0, acc=0.8)
    tracker.update(loss=0.9, acc=0.85)
    print(f"Average loss: {tracker.get_average('loss'):.3f}")

    print("✅ 工具測試完成")