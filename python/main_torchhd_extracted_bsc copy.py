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
    """
    training_patients = ["chb"+str(x).zfill(2) for x in range(8, 8+1)]
    # training_features = ["feature"+str(x).zfill(2) for x in range(1, 4+1)]
    architecture = [0]
    training_features = [1]
    for arch in architecture:
        if arch == 0:
            archi = "BSC"
        for feat in training_features:
            if feat == 1:
                memory = [np.loadtxt("memory/feature1/BSC_0.txt", int),np.loadtxt("memory/feature1/BSC_1.txt", int)]
            # for i in range(len(memory)):
                # mem = np.array(memory[i])
                # np.savetxt(f"memory/feature{feat}/{archi}_{i}.txt",mem)
            patientn = 0
            for patient in training_patients:
                for result in ["seizures", "non-seizures"]:
                    files_patient = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                    for exc in files_patient:
                        outer_counter = 0
                        files_patient2 = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                        files_count = len(files_patient2)-1
                        tot_arr = np.zeros((n), dtype=int)
                        for file in files_patient2:
                            if exc == file: 
                                continue
                            else:
                                path = f"features/perfile/feature{feat}/{patient}/{result}/{file}"
                                feat_arr = np.loadtxt(path, int)
                                if feat == 1:
                                    shape = int(len(feat_arr[0])/17)
                                feat_arr = feat_arr.reshape(-1,shape)
                                print(feat_arr)
                                print(len(feat_arr))
                                wins = int(len(feat_arr)/17)
                                print(wins)
                                
                                for i in range(wins):
                                    window = feat_arr[i*17:(i+1)*17].transpose()
                                    window = window[1:]
                                    win_arr = np.zeros((len(window),n), dtype=int)
                                    wincount = 0
                                    for samp in window:
                                        # print(samp)
                                        samp_arr = np.zeros((num_chs,n), dtype=int)
                                        for j in range(num_chs):
                                            samp_arr[j] = hdc.bind(memory[1][int(samp[j])],memory[0][j])
                                        win_arr[wincount] = hdc.bundle(samp_arr)
                                        wincount+=1
                                    tot_arr = hdc.bundle_cont(hdc.bundle(win_arr),tot_arr)
                                    print(tot_arr)
                                    print(f"{i+1} out of {wins}")
                                print(f"tot_arr {tot_arr}")
                                print(f"{outer_counter+1} out of {files_count}")
                                outer_counter += 1
                        res = hdc.binarize_cont(tot_arr)
                        outp = np.array(res)
                        filename_out = f"assocmem/{archi}/feature{feat}/{patient}/{result}_{exc[:-9]}.txt"   
                        np.savetxt(filename_out, outp)
                        print(f"{exc} success")
                patientn += 1
    # """
    # TESTING
    
    patientn = 5
    testing_patients = ["chb"+str(x).zfill(2) for x in range(patientn, 7+1)]
    architecture = [0]
    testing_features = [1]
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open("hdc_testing_log.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Testing HDC Begin {log_time}
                ---
                #1 - LBP
                #2 - Spectral Power

                """
            ))
    for arch in architecture:
        if arch == 0:
            archi = "BSC"
        with open("hdc_testing_log.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Now Testing {archi} Architecture
                ---

                """
            ))
        for feat in testing_features:
            if feat == 1:
                memory = [np.loadtxt("memory/feature1/BSC_0.txt", int),np.loadtxt("memory/feature1/BSC_1.txt", int)]
            patientn -= 1
            for patient in testing_patients:
                with open("hdc_testing_log.txt", 'a') as log:
                            log.write(dedent(
                                f"""
                                ---
                                {patient}
                                ---"""
                            ))
                for result in ["seizures", "non-seizures"]:
                    files_patient = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                    files_count = len(files_patient)-1
                    counter = 0
                    for exc in files_patient:
                        
                        path = f"features/perfile/feature{feat}/{patient}/{result}/{exc}"
                        if result == "seizures":
                            if arch == 0:
                                am = [np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/seizures_{exc[:-9]}.txt", int),np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/non-seizures_{exc[:-11]}.txt", int)]
                        else:
                            if arch == 0:
                                am = [np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/seizures_{exc[:-9]}_0.txt",int),np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/non-seizures_{exc[:-9]}.txt",int)]
                        feat_arr = np.loadtxt(path)
                        if feat == 1:
                            shape = int(len(feat_arr[0])/17)
                        feat_arr = feat_arr.reshape(-1,shape)
                        wins = int(len(feat_arr)/17)
                        print(wins)
                        seizcount = 0
                        nonseizcount = 0
                        for i in range(wins):
                            window = feat_arr[i*17:(i+1)*17].transpose()
                            window = window[1:]
                            win_arr = np.zeros((len(window),n), dtype=int)
                            wincount = 0
                            for samp in window:
                                # print(samp)
                                samp_arr = np.zeros((num_chs,n))
                                for j in range(num_chs):
                                    samp_arr[j] = hdc.bind(memory[1][int(samp[j])],memory[0][j])
                                win_arr[wincount] = hdc.bundle(samp_arr)
                                wincount += 1
                            res = hdc.bundle(win_arr)
                            # print(res)
                            if arch == 0:
                                seiz = hdc.compute_similarity(res, am[0])
                                nonseiz = hdc.compute_similarity(res, am[1])
                                print(f"win {i+1} of {wins}")
                                if seiz < nonseiz:
                                    seizcount+=1
                                    print("seiz")
                                else:
                                    nonseizcount+=1
                                    print("nonseiz")
                                print(f"{seiz} vs {nonseiz}")
                        if result == "seizures":
                            sens = seizcount/wins
                        else:
                            spec = nonseizcount/wins
                        log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                        with open("hdc_testing_log.txt", 'a') as log:
                                log.write(dedent(
                                    f"""
                                    {exc[6:-9]}, {seizcount}, {nonseizcount}"""
                                ))
                        print(f"seizure sim: {seiz}, nonseizure sim: {nonseiz}")
                        print(f"{counter+1} out of {files_count}")
                        counter += 1
                patientn+=1
    


                                        
                                            
