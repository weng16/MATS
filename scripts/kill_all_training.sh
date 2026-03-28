#!/bin/bash
# 停掉所有 MATS 训练相关进程
echo "Killing all scripts/train.py processes..."
pkill -9 -f "scripts/train.py" 2>/dev/null
pkill -9 -f "run_all_gpu" 2>/dev/null
sleep 2
if pgrep -f "scripts/train.py" >/dev/null; then
    echo "Some processes still running, force killing..."
    for pid in $(pgrep -f "scripts/train.py"); do kill -9 $pid 2>/dev/null; done
    sleep 1
fi
pgrep -f "scripts/train.py" >/dev/null && echo "Warning: some processes may still be running" || echo "All training processes stopped."
