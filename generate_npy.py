import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_npy_from_csv(dataset_name, data_dir, train_ratio=0.7, val_ratio=0.1):
    csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(csv_path):
        # 尝试小写的文件名 (针对 Weather, Traffic, Electricity)
        csv_path = os.path.join(data_dir, f"{dataset_name.lower()}.csv")
        if not os.path.exists(csv_path):
            print(f"CSV not found: {os.path.join(data_dir, f'{dataset_name}.csv')} or {csv_path}")
            return
        
    print(f"Processing {dataset_name}...")
    df = pd.read_csv(csv_path)
    
    # Remove date column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        
    data = df.values.astype(np.float32)
    # basicts expects shape [L, C, 1]
    data = np.expand_dims(data, axis=-1)
    
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = data[:train_len]
    val_data = data[train_len:train_len+val_len]
    test_data = data[train_len+val_len:]
    
    print(f"Total: {total_len}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    np.save(os.path.join(data_dir, "train_data.npy"), train_data)
    np.save(os.path.join(data_dir, "val_data.npy"), val_data)
    np.save(os.path.join(data_dir, "test_data.npy"), test_data)
    print(f"Saved .npy files to {data_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "data"
    for ds in ["ETTm1", "ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic"]:
        ds_dir = base_dir / ds
        if ds_dir.exists():
            generate_npy_from_csv(ds, str(ds_dir))