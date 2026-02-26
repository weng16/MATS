#!/usr/bin/env python3
"""
一键下载 Autoformer 官方数据集并解压到 data/ 目录
支持：ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

data_root = Path(__file__).resolve().parent.parent / "data"
data_root.mkdir(exist_ok=True)

datasets = {
    "ETTh1": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "ETTh2": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "ETTm1": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "ETTm2": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "Weather": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "ECL": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
    "Traffic": "https://drive.google.com/uc?id=1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy",
}

def download_from_gdrive(file_id, save_path):
    """使用 gdown 下载 Google Drive 文件"""
    subprocess.check_call(["gdown", file_id, "-O", str(save_path)])

def extract_zip(zip_path, extract_to):
    """解压 zip 文件"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

def main():
    print("📦 开始下载 Autoformer 官方数据集...")
    for ds_name, url in datasets.items():
        ds_dir = data_root / ds_name
        ds_dir.mkdir(exist_ok=True)
        zip_path = ds_dir / f"{ds_name}.zip"
        
        # 若已存在 csv 则跳过
        if (ds_dir / f"{ds_name}.csv").exists():
            print(f"✅ {ds_name}.csv 已存在，跳过")
            continue
        
        print(f"⬇️  正在下载 {ds_name}...")
        file_id = url.split("id=")[1]
        download_from_gdrive(file_id, zip_path)
        
        print(f"📂 正在解压 {ds_name}.zip...")
        extract_zip(zip_path, ds_dir)
        
        # 删除 zip 节省空间
        zip_path.unlink()
        print(f"✅ {ds_name} 完成！")
    
    print("🎉 所有数据集已就绪，目录结构：")
    subprocess.run(["tree", "-L", "2", str(data_root)])

if __name__ == "__main__":
    main()