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

def create_memory(n: int, mem_type: int, items: int = None) -> None:
    """
    Creates a .npy file in "training-data" subdirectory containing the memory of chosen type.

    ### Parameters:

        n : int
            Dimension of hypervector

        mem_type : int
                0 = Channel memory;
                1 = LBP;  
                2 = Spectral Power;  
                3 = Mean amplitude;  
                4 = Line length;  
                5 = Feature IDs

        items: int
            Number of items in memory.
            For LBP, this dictates the number of bits, i.e. 2**items = total number of patterns.
    """
    if mem_type == 0: # Channel memory
        memory_chs = hdc.generate_randommemory(n=n, p=0.5, d=items)
        with open("training-data/memory_channels.npy", 'wb') as f:
            np.save(f, memory_chs)
    elif mem_type == 1: # LBP
        memory_LBP = hdc.generate_randommemory(n=n, p=0.5, d=2**items)
        with open("training-data/memory_LBP.npy", 'wb') as f:
            np.save(f, memory_LBP)
    elif mem_type == 2: # Spectral Power
        memory_spi = hdc.generate_randommemory(n=n, p=0.5, d=items)
        memory_spv = hdc.generate_linearmemory(n=n, p=0.5, d=items)
        with open("training-data/memory_sp.npy", 'wb') as f:
            np.save(f, memory_spi)
            np.save(f, memory_spv)
    elif mem_type == 3: # Mean amplitude
        memory_ma = hdc.generate_linearmemory(n=n, p=0.5, d=items)
        with open("training-data/memory_ma.npy", 'wb') as f:
            np.save(f, memory_ma)
    elif mem_type == 4: # Line length
        memory_ll = hdc.generate_linearmemory(n=n, p=0.5, d=items)
        with open("training-data/memory_ll.npy", 'wb') as f:
            np.save(f, memory_ll)
    elif mem_type == 5: # Feature IDs
        memory_FID = hdc.generate_randommemory(n=n, p=0.5, d=4)
        with open("training-data/memory_FID.npy", 'wb') as f:
            np.save(f, memory_FID)
    

def get_memory(mem_type: int) -> np.ndarray:
    """
    Returns the memory as a numpy array from a .npy file of the chosen type.

    ### Parameters:
        mem_type : int
                0 = Channel memory;
                1 = LBP;  
                2 = Spectral Power;  
                3 = Mean amplitude;  
                4 = Line length;  
                5 = Feature IDs
    """
    if mem_type == 0: # Channels
        with open ("training-data/memory_channels.npy", 'rb') as f:
            memory_chs = np.load(f)
        return memory_chs
        
    elif mem_type == 1: # LBP only
        with open ("training-data/memory_LBP.npy", 'rb') as f:
            memory_LBP = np.load(f)
        return memory_LBP
        
    elif mem_type == 2: # Spectral Power only
        with open ("training-data/memory_sp.npy", 'rb') as f:
            memory_spi = np.load(f)
            memory_spv = np.load(f)
        return [memory_spi, memory_spv]
        
    elif mem_type == 3: # Mean amplitude
        with open ("training-data/memory_ma.npy", 'rb') as f:
            memory_ma = np.load(f)
        return memory_ma
        
    elif mem_type == 4: # Line length
        with open ("training-data/memory_ll.npy", 'rb') as f:
            memory_ll = np.load(f)
        return memory_ll
        
    elif mem_type == 5: # Feature IDs
        with open("training-data/memory_FID.npy", 'rb') as f:
            memory_FID = np.load(f)
        return memory_FID
        

def extract_npy(filepath: str):
    """
    Extracts and returns data of a patient file given a filepath.

    ### Parameters:
        filepath: str
            Filepath of patient file relative to current directory.
        
    ### Returns:
        label_chs: numpy array of strings
            Array of channel names in current patient file.

        value_chs: numpy array of array of floats
            Arrays of sample values inside an array of different channels in the patient file.

        last_samp: int
            Index of the last sample in all channel sample values.

        samp_freq: int
            Sampling frequency of the patient file.

    """
    with open(filepath, 'rb') as f:
        label_chs = np.load(f)
        value_chs = np.load(f)
        last_samp = np.load(f)
        samp_freq = np.load(f) 
    
    return label_chs, value_chs, int(last_samp), int(samp_freq)

if __name__ == "__main__":
    n = 10000
    window_size = 256 # 1.0 seconds
    window_step = 128 # 0.5 seconds
    fs = 256
    d = 6 # LBP bit size
    num_chs = 17 # constant
    levels = 128
    
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open("hdc_training_log.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Training HDC Begin {log_time}
                ---
                #1 - LBP
                #2 - Spectral Power
                #3 - LBP + LL + MA
                #4 - SP + LL

                """
            ))

    ###################################################################

    # ITEM MEMORY CREATION (comment out as needed)
    # create_memory(n=n, mem_type=0, items = num_chs) # channels
    # create_memory(n=n, mem_type=1, items = d) # LBP
    # create_memory(n=n, mem_type=2, items = levels) # spectral power
    # create_memory(n=n, mem_type=3, items = levels) # mean amplitude
    # create_memory(n=n, mem_type=4, items = levels) # line length
    # create_memory(n=n, mem_type=5) # feature IDs
    
    ###################################################################

    # TRAINING

    training_patients = ["chb"+str(x).zfill(2) for x in range(1, 24+1)]
    # training_features = ["feature"+str(x).zfill(2) for x in range(1, 4+1)]
    training_features = [2]
    for feat in training_features:
        if feat == 1:
            memory = [get_memory(0), get_memory(1)]
            shape = 65
        elif feat == 2:
            memory = [get_memory(0), get_memory(2)]
            shape = 7
            min_sp = np.loadtxt("minmax/sp_min.txt")
            max_sp = np.loadtxt("minmax/sp_max.txt")
        elif feat == 3:
            memory = [get_memory(0), get_memory(4), get_memory(3), get_memory(1), get_memory(5)]
            shape = 67
        elif feat == 4:
            memory = [get_memory(0), get_memory(2), get_memory(4), get_memory(5)]
            min_sp = np.loadtxt("minmax/sp_min.txt")
            max_sp = np.loadtxt("minmax/sp_max.txt")
            shape = 8
        for patient in training_patients:
            for result in ["seizures", "non-seizures"]:
                files_patient = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                for exc in files_patient:
                    seizure_hv = np.zeros(n)
                    nonseizure_hv = np.zeros(n)
                    for file in files_patient:
                        if exc == file:
                            continue
                        else:
                            path = f"features/perfile/feature{feat}/{patient}/{result}/{file}"
                            feat_arr = np.loadtxt(path)
                            feat_arr = feat_arr.reshape(-1,shape)
                            for window in feat_arr:
                                feat_len = len(window)
                                for i in range(feat_len):
                                    if i==0:
                                        channel = memory[0][int(window[0])-1]
                                    else:
                                        if feat == 1:
                                            continue
                                        elif feat == 2:
                                            sp_id = memory[1][0][i-1]
                                            sp_min = min_sp[i-1]
                                            sp_max = max_sp[i-1]
                                            sp_val = hdc.extract_linearMemory(memory[1][1],window[i],sp_min, sp_max, levels)
                                            print(f"{i} ok")
                                            res = hdc.bind(sp_id, sp_val)
                                            
                                        elif feat == 3:
                                            continue
                                        elif feat == 4:
                                            continue
                                    
                                        
