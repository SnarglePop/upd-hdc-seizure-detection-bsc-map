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
import random

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
    with open("svm_extracted_testlog.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Training SVM Begin {log_time}
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

    training_patients = list(range(1,24+1))

    training_features = [1,2,3,4]
    for feat in training_features:
        log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        with open("svm_extracted_testlog.txt", 'a') as log:
                log.write(dedent(
                    f"""
                    ---

                    Starting Feature {feat} at {log_time}
                    
                    """
                ))
        for exc in training_patients:
            log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            with open("svm_extracted_testlog.txt", 'a') as log:
                    log.write(dedent(
                        f"""
                        Now excluding {exc} at {log_time}

                        """
                    ))
            feature_array = np.array([])
            label_array = np.array([])
            feature_array_test = np.array([])
            label_array_test = np.array([])
            if exc<10:
                file_feat = f"features/feature{feat}/chb0{exc}_feat.txt"
                file_label = f"features/feature{feat}/chb0{exc}_label.txt"
            else:
                file_feat = f"features/feature{feat}/chb{exc}_feat.txt"
                file_label = f"features/feature{feat}/chb{exc}_label.txt"
            feat_arr = np.loadtxt(file_feat)
            label_arr = np.loadtxt(file_label)
            total = len(feat_arr)
            exclusion = list(range(0,total))
            removal = int(total*0.3)
            exclusion = random.choices(exclusion, k=removal)
            print(exclusion)
            label_count = -1
            for i in range(0,total):
                if feat == 1:
                    featlen = 65
                if feat == 2:
                    featlen = 7
                if feat == 3:
                    featlen = 67
                if feat == 4:
                    featlen = 8
                if i in exclusion:
                    working = np.array(feat_arr[i]).reshape(-1, featlen)
                    for x in working:
                        label_count += 1
                        feature_array_test = np.append(feature_array_test, x)
                        label_array_test = np.append(label_array_test, label_arr[label_count])
                else:
                    working = np.array(feat_arr[i]).reshape(-1,featlen)
                    for x in working:
                        
                        label_count += 1

                        feature_array = np.append(feature_array, x)
                        label_array = np.append(label_array, label_arr[label_count])
                        print(f"Now on {label_count} of {len(label_arr)} ({len(feature_array)} vs {len(label_array)}) for patient {exc} at feature {feat}")
            if feat == 1:
                feature_array = feature_array.reshape(-1,65)
                feature_array_test = feature_array_test.reshape(-1,65)
            if feat == 2:
                feature_array = feature_array.reshape(-1,7)
                feature_array_test = feature_array_test.reshape(-1,7)
            if feat == 3:
                feature_array = feature_array.reshape(-1,67)
                feature_array_test = feature_array_test.reshape(-1,67)
            if feat == 4:
                feature_array = feature_array.reshape(-1,8)
                feature_array_test = feature_array_test.reshape(-1,8)
            print(len(feature_array))
            print(len(label_array))
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(feature_array, label_array)
            print(f"Completed training leave one out for chb{exc} with length of {len(feature_array)}")
            log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            sens = 0
            spec = 0
            sens_tot = 0
            spec_tot = 0
            test_out = clf.predict(feature_array_test)
            np.savetxt(f"predictions/extracted/SVM_Feature{feat}_chb{exc}.txt", test_out)
            for l in range(len(test_out)):
                t = test_out[l]
                if label_array_test[l] == -1:
                    spec_tot += 1
                    if t == -1:
                        spec += 1
                else:
                    sens_tot += 1
                    if t == 1:
                        sens += 1
            print(f"Seizures: {sens} out of {sens_tot}")
            avg_sens = sens/sens_tot
            print(f"Non-seizures: {spec} out of {spec_tot}")
            avg_spec = spec/spec_tot
            with open("svm_extracted_testlog.txt", 'a') as log:
                log.write(dedent(
                    f"""

                    chb{exc} at feature {feat}
                    Sensitivity: {avg_sens} ({sens} out of {sens_tot})
                    Specificity: {avg_spec} ({spec} out of {spec_tot})

                    """
                ))
        log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        with open("svm_extracted_testlog.txt", 'a') as log:
                log.write(dedent(
                    f"""
                    ---

                    SVM Feature {feat} End {log_time}
                    
                    ---
                    """
                ))
                
            
