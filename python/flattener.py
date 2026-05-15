import os, time, re
import numpy as np
import hdc_methods as hdc
from textwrap import dedent
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import os, time, re
import numpy as np
import hdc_methods as hdc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from datetime import datetime
import scipy.signal
import torch, torchhd 

if __name__ == "__main__":
    """
    folders = [1,2,3,4]
    folderin = "features/consolidated/"
    labels = ["chb"+str(x).zfill(2) for x in range(1, 24+1)]
    for i in range(len(labels)):
        labels[i] = labels[i]+"_label.txt"
    for fold in folders:
        cur_fold = folderin+f"feature{fold}/"
        files = os.listdir(cur_fold)
        for file in files:
            print(file)
            if file in labels:
                working_array = np.loadtxt(cur_file)
                filename = f"features/feature{fold}/{file}"
                np.savetxt(filename, working_array)
                continue
            else:
                cur_file = cur_fold+file
                working_array = np.loadtxt(cur_file)
                print(len(working_array))
                working_array = working_array.flatten()
                print(len(working_array))
                if fold == 1:
                    working_array = working_array.reshape(-1,65*17)
                elif fold == 2:
                    working_array = working_array.reshape(-1,7*17)
                elif fold == 3:
                    working_array = working_array.reshape(-1,67*17)
                elif fold == 4:
                    working_array = working_array.reshape(-1,8*17)
                print(len(working_array))
                print(working_array[0])
                print(working_array[1])
                filename = f"features/feature{fold}/{file}"
                np.savetxt(filename, working_array)
                # working_array = np.loadtxt(filename)
                # print(working_array[0])
                # print(working_array[1])
    """
    a = torchhd.BSCTensor.random(1,10000)[0]
    b = np.array(a)
    print(b)
    print("ok")