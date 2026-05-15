import os, time, re
import numpy as np
import hdc_methods as hdc
from textwrap import dedent
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

with open("minmax_sp.npy", 'rb') as f:
    min_sp = np.load(f)
    max_sp = np.load(f)
print(len(min_sp))
for i in min_sp:
    print(i)
"""
sp_min = np.array([])
sp_max = np.array([])
for i in range(6):
    tempmin = np.array([])
    tempmax = np.array([])
    for p in min_sp:
        tempmin = np.append(tempmin,p[i])
    sp_min = np.append(sp_min,np.min(tempmin))
    for p in max_sp:
        tempmax = np.append(tempmax,p[i])
    sp_max = np.append(sp_max,np.max(tempmax))
filename_min = "minmax/sp_min.txt"
filename_max = "minmax/sp_max.txt"
np.savetxt(filename_min,sp_min)
np.savetxt(filename_max,sp_max)
"""