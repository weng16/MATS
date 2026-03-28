import numpy as np
import glob
for f in glob.glob("data/*/*.npy"):
    try:
        data = np.load(f, mmap_mode="r")
        print(f"{f}: {data.shape}")
    except Exception as e:
        print(f"{f}: {e}")