"""
时序预测数据集模块

支持常用长时序预测(Long-term Time Series Forecasting)数据集:
- ETT (ETTh1, ETTh2, ETTm1, ETTm2) - 电力变压器温度
- Weather - 天气数据
- Traffic - 交通流量
- Electricity (ECL) - 电力消耗
- Exchange - 汇率
- ILI - 流感数据

数据处理:
- 标准化 (StandardScaler)
- 滑窗采样
- 训练/验证/测试划分 (7:1:2)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import os
import warnings


# ==================== 数据集配置 ====================
DATASET_CONFIG = {
    # ETT数据集
    'ETTh1': {'file': 'ETTh1.csv', 'T': 'OT', 'freq': 'h', 'features': 7},
    'ETTh2': {'file': 'ETTh2.csv', 'T': 'OT', 'freq': 'h', 'features': 7},
    'ETTm1': {'file': 'ETTm1.csv', 'T': 'OT', 'freq': 't', 'features': 7},
    'ETTm2': {'file': 'ETTm2.csv', 'T': 'OT', 'freq': 't', 'features': 7},
    # 其他数据集
    'Weather': {'file': 'weather.csv', 'T': 'OT', 'freq': 't', 'features': 21},
    'Traffic': {'file': 'traffic.csv', 'T': 'OT', 'freq': 'h', 'features': 862},
    'Electricity': {'file': 'electricity.csv', 'T': 'OT', 'freq': 'h', 'features': 321},
    'ECL': {'file': 'electricity.csv', 'T': 'OT', 'freq': 'h', 'features': 321},  # 别名
    'Exchange': {'file': 'exchange_rate.csv', 'T': 'OT', 'freq': 'd', 'features': 8},
    'ILI': {'file': 'national_illness.csv', 'T': 'OT', 'freq': 'w', 'features': 7},
}


class StandardScaler:
    """标准化工具"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1  # 避免除零
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
            std = torch.tensor(self.std, device=data.device, dtype=data.dtype)
            return data * std + mean
        return data * self.std + self.mean
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


class TimeSeriesDataset(Dataset):
    """
    通用时序预测数据集
    
    支持:
    - 滑窗采样
    - 多变量(M)/单变量(S)/多变量预测单变量(MS)
    - 时间特征编码
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 96,
        label_len: int = 48,
        features: str = 'M',
        target_idx: int = -1,
        scaler: Optional[StandardScaler] = None,
        time_features: Optional[np.ndarray] = None,
        mode: str = 'train'
    ):
        """
        参数:
            data: [T, D] 时序数据
            seq_len: 输入序列长度 (历史窗口)
            pred_len: 预测序列长度 (预测窗口)
            label_len: 标签序列长度 (用于decoder的起始token)
            features: 'M'多变量预测多变量 / 'S'单变量 / 'MS'多变量预测单变量
            target_idx: 目标列索引 (用于MS模式)
            scaler: 标准化器
            time_features: 时间特征 [T, num_time_features]
            mode: 'train', 'val', 'test'
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        self.target_idx = target_idx
        self.mode = mode
        
        # 数据处理
        self.data_raw = data
        
        # 标准化
        if scaler is not None:
            self.scaler = scaler
            self.data = scaler.transform(self.data_raw)
        else:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data_raw)
        
        # 时间特征
        self.time_features = time_features
        
        # 转为tensor
        self.data = torch.from_numpy(self.data).float()
        if self.time_features is not None:
            self.time_features = torch.from_numpy(self.time_features).float()
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 输入序列
        seq_x = self.data[s_begin:s_end]
        
        # 目标序列 (decoder输入)
        seq_y = self.data[r_begin:r_end]
        
        # 预测目标
        if self.features == 'MS':
            target = self.data[s_end:s_end + self.pred_len, self.target_idx:self.target_idx+1]
        else:
            target = self.data[s_end:s_end + self.pred_len]
        
        result = {
            'x': seq_x,           # [seq_len, D] 输入
            'y': target,          # [pred_len, D] 预测目标
            'y_full': seq_y,      # [label_len + pred_len, D] decoder输入
        }
        
        # 时间特征
        if self.time_features is not None:
            result['x_mark'] = self.time_features[s_begin:s_end]
            result['y_mark'] = self.time_features[r_begin:r_end]
        
        return result


def time_features_from_dates(dates: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """
    从日期提取时间特征
    
    返回: [T, num_features] 时间特征
    """
    features = []
    
    # 小时
    if freq in ['h', 't']:
        features.append(dates.hour / 23.0 - 0.5)
    
    # 星期几
    features.append(dates.dayofweek / 6.0 - 0.5)
    
    # 月中第几天
    features.append((dates.day - 1) / 30.0 - 0.5)
    
    # 年中第几周
    features.append((dates.isocalendar().week.values - 1) / 52.0 - 0.5)
    
    # 月份
    features.append((dates.month - 1) / 11.0 - 0.5)
    
    return np.stack(features, axis=1).astype(np.float32)


def load_dataset(
    root_path: str,
    dataset_name: str,
    features: str = 'M'
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], int]:
    """
    加载数据集
    
    参数:
        root_path: 数据根目录
        dataset_name: 数据集名称
        features: 特征模式
        
    返回:
        data: [T, D] 数据
        time_features: [T, F] 时间特征
        columns: 列名列表
        target_idx: 目标列索引
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIG.keys())}")
    
    config = DATASET_CONFIG[dataset_name]
    file_path = os.path.join(root_path, config['file'])
    
    if not os.path.exists(file_path):
        warnings.warn(f"Dataset file not found: {file_path}. Generating synthetic data.")
        num_features = config['features']
        data = generate_synthetic_data(2000, num_features)
        columns = ['OT'] + [f'feature_{i}' for i in range(num_features - 1)]
        return data, None, columns, 0
    
    # 读取CSV
    df = pd.read_csv(file_path)
    
    # 处理日期列
    time_features = None
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        time_features = time_features_from_dates(dates, config['freq'])
        df = df.drop('date', axis=1)
    
    columns = df.columns.tolist()
    data = df.values.astype(np.float32)
    
    # 目标列索引
    target_idx = columns.index(config['T']) if config['T'] in columns else 0
    
    return data, time_features, columns, target_idx


def generate_synthetic_data(
    length: int = 2000,
    num_features: int = 7,
    seed: int = 42
) -> np.ndarray:
    """
    生成合成时序数据用于测试
    
    包含多种模式: 周期性、趋势、噪声、突变点
    """
    np.random.seed(seed)
    
    t = np.arange(length)
    data = np.zeros((length, num_features))
    
    for i in range(num_features):
        # 基础周期
        period = 24 * (1 + i % 3)  # 24, 48, 72小时周期
        data[:, i] = np.sin(2 * np.pi * t / period)
        
        # 添加趋势
        if i % 3 == 1:
            data[:, i] += 0.001 * t
        
        # 添加噪声
        data[:, i] += 0.1 * np.random.randn(length)
        
        # 添加突变
        if i == 0:
            change_points = [length // 3, 2 * length // 3]
            for cp in change_points:
                data[cp:, i] += np.random.randn() * 0.5
    
    # 最后一列作为目标，是其他特征的组合
    data[:, -1] = 0.3 * data[:, 0] + 0.3 * data[:, 1] + 0.2 * data[:, 2] + 0.2 * np.random.randn(length)
    
    return data.astype(np.float32)


def create_dataloaders(
    root_path: str,
    dataset_name: str = 'ETTh1',
    seq_len: int = 96,
    pred_len: int = 96,
    label_len: int = 48,
    features: str = 'M',
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, Dict]:
    """
    创建训练/验证/测试DataLoader
    
    参数:
        root_path: 数据根目录
        dataset_name: 数据集名称
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        label_len: 标签序列长度
        features: 'M'/'S'/'MS'
        batch_size: 批大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: DataLoader工作线程数
        
    返回:
        train_loader, val_loader, test_loader, scaler, info_dict
    """
    # 加载数据
    data, time_features, columns, target_idx = load_dataset(
        root_path, dataset_name, features
    )
    
    total_len = len(data)
    
    # 划分数据 (7:1:2)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    train_time = time_features[:train_end] if time_features is not None else None
    val_time = time_features[train_end:val_end] if time_features is not None else None
    test_time = time_features[val_end:] if time_features is not None else None
    
    # 在训练集上拟合scaler
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(
        train_data, seq_len, pred_len, label_len, features, target_idx,
        scaler=scaler, time_features=train_time, mode='train'
    )
    val_dataset = TimeSeriesDataset(
        val_data, seq_len, pred_len, label_len, features, target_idx,
        scaler=scaler, time_features=val_time, mode='val'
    )
    test_dataset = TimeSeriesDataset(
        test_data, seq_len, pred_len, label_len, features, target_idx,
        scaler=scaler, time_features=test_time, mode='test'
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    # 数据信息
    info = {
        'dataset': dataset_name,
        'num_features': data.shape[1],
        'target_idx': target_idx,
        'columns': columns,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'seq_len': seq_len,
        'pred_len': pred_len
    }
    
    return train_loader, val_loader, test_loader, scaler, info


# 兼容旧接口
def create_dataloader(*args, **kwargs):
    """旧接口兼容"""
    return create_dataloaders(*args, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TimeSeriesDataset")
    print("=" * 60)
    
    # 测试合成数据
    print("\n1. Testing with synthetic data:")
    data = generate_synthetic_data(2000, 7)
    print(f"   Data shape: {data.shape}")
    
    train_loader, val_loader, test_loader, scaler, info = create_dataloaders(
        root_path='./data',
        dataset_name='ETTh1',
        seq_len=96,
        pred_len=96,
        batch_size=32
    )
    
    print(f"\n2. Dataset info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print(f"\n3. DataLoader sizes:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 检查一个batch
    batch = next(iter(train_loader))
    print(f"\n4. Batch shapes:")
    print(f"   x: {batch['x'].shape}")
    print(f"   y: {batch['y'].shape}")
    print(f"   y_full: {batch['y_full'].shape}")
    
    print("\n" + "=" * 60)
    print("Available datasets:", list(DATASET_CONFIG.keys()))
    print("=" * 60)
