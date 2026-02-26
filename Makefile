#!/usr/bin/env python3
"""
一键 Makefile
make main     # 跑主实验
make ablation # 跑消融实验
make clean    # 清理中间结果
"""
.PHONY: main ablation clean

main:
	python scripts/run_main.py

ablation:
	python scripts/run_ablation.py

clean:
	rm -rf results/*/.*hydra/ outputs/temp/ .hydra/ __pycache__/