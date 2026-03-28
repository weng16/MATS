#!/usr/bin/env python3
"""
批量跑消融实验（ablation）
只在代表性数据集上跑，pred_len=96
"""
import subprocess

abl_datasets = ["ETTh1", "ETTm1"]   # Table 3
abl_types  = ["no_prototype", "no_causal", "no_verification", "hard_routing", "single_expert"]
pred_len   = 96

print("=========================================")
print("开始消融实验（Ablation Study）")
print(f"数据集: {abl_datasets}")
print(f"消融类型: {abl_types}")
print("=========================================")

for ds in abl_datasets:
    for abl in abl_types:
        print(f"\n>>> 消融: {abl} on {ds}")
        cfg = f"ablation/{abl}/{ds.lower()}_{pred_len}"
        subprocess.run(["python", "scripts/train.py", f"--config-name={cfg}"])

print("\n=========================================")
print("✅ 所有消融实验完成！")
print("=========================================")