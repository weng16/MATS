#!/usr/bin/env python3
"""
批量跑主实验（main）
数据集 × 预测长度 全组合
"""
import subprocess
import itertools

datasets = ["ETTh1", "ETTh2", "Weather", "ECL", "Traffic"]
pred_lens = [96, 192, 336, 720]

print("=========================================")
print("开始主实验（Main Experiments）")
print(f"总计: {len(datasets)} 数据集 × {len(pred_lens)} 预测长度 = {len(datasets)*len(pred_lens)} 组实验")
print("=========================================")

for ds, pl in itertools.product(datasets, pred_lens):
    print(f"\n>>> 运行: {ds} - {pl}")
    cfg = f"main/{ds.lower()}_{pl}"
    subprocess.run(["python", "scripts/train.py", f"--config-name={cfg}"])

print("\n=========================================")
print("✅ 所有主实验完成！")
print("=========================================")