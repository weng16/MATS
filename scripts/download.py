#!/usr/bin/env python3
"""
时序预测常用数据集下载脚本
支持：ETT系列、Weather、Electricity、Traffic等
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 数据集配置：(数据集名称, 文件名, URL)
DATASETS: List[Tuple[str, str, str]] = [
    # ETT 系列
    ("ETTh1", "ETTh1.csv", 
     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"),  # 删掉末尾空格
    ("ETTh2", "ETTh2.csv", 
     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"),  # 删掉末尾空格
    ("ETTm1", "ETTm1.csv", 
     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"),  # 删掉末尾空格
    ("ETTm2", "ETTm2.csv", 
     "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"),  # 删掉末尾空格
    
    # Weather
    ("Weather", "weather.csv",
     "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv"),  # 删掉末尾空格
    
    # Electricity
    ("Electricity", "electricity.csv",
     "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity/electricity.csv"),  # 删掉末尾空格
    
    # Traffic
    ("Traffic", "traffic.csv",
     "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/traffic/traffic.csv"),  # 删掉末尾空格
    
    # 可选数据集
    ("ExchangeRate", "exchange_rate.csv",
     "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/exchange_rate/exchange_rate.csv"),  # 删掉末尾空格
    ("ILI", "national_illness.csv",
     "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/illness/national_illness.csv"),  # 删掉末尾空格
]


def download_file(url: str, output_path: Path, desc: str = "") -> bool:
    """下载文件并显示进度"""
    try:
        print(f"  → 下载 {desc}...", end=" ", flush=True)
        
        def reporthook(count, block_size, total_size):
            """进度回调"""
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  → 下载 {desc}... {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        
        # 验证文件大小
        file_size = output_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"\r  ✓ {desc} 下载完成 ({size_mb:.2f} MB)    ")
        return True
        
    except Exception as e:
        print(f"\r  ✗ {desc} 下载失败: {e}")
        return False


def main():
    print("=" * 50)
    print("时序预测数据集下载工具")
    print(f"数据存储目录: {DATA_DIR}")
    print("=" * 50)
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, (dataset_name, filename, url) in enumerate(DATASETS, 1):
        print(f"\n📦 [{i}/{len(DATASETS)}] {dataset_name}")
        
        # 创建数据集目录
        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # 输出文件路径
        output_file = dataset_dir / filename
        
        # 跳过已存在的文件
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)
            print(f"  ⚠ 文件已存在，跳过 ({file_size:.2f} MB)")
            success_count += 1
            continue
        
        # 下载
        if download_file(url, output_file, filename):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("\n" + "=" * 50)
    print("✅ 下载完成！")
    print("=" * 50)
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    print("\n数据集清单:")
    
    for dataset_name, filename, _ in DATASETS[:7]:  # 只显示主要数据集
        file_path = DATA_DIR / dataset_name / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  • {dataset_name:12s}: {file_path} ({size_mb:.2f} MB)")
    
    print("\n💡 提示：如需重新下载，请删除对应文件后重新运行")


if __name__ == "__main__":
    main()