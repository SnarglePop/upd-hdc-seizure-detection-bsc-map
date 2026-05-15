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
    n = 10000
    window_size = 256 # 1.0 seconds
    window_step = 128 # 0.5 seconds
    fs = 256
    d = 6 # LBP bit size
    num_chs = 17 # constant
    levels = 128
    
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open("hdc_training_log_3.txt", 'a') as log:
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
    # """
    training_patients = ["chb"+str(x).zfill(2) for x in range(1, 5+1)]
    # training_features = ["feature"+str(x).zfill(2) for x in range(1, 4+1)]
    architecture = [0]
    training_features = [1]
    for arch in architecture:
        if arch == 0:
            archi = "BSC"
        elif arch == 1:
            archi = "MAP"
        elif arch == 2:
            archi = "BSBC"
        for feat in training_features:
            if feat == 1:
                if arch == 0:
                    memory = [torchhd.BSCTensor(np.loadtxt("memory/feature1/BSC_0.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature1/BSC_1.txt"))]
                elif arch == 1:
                    memory = [torchhd.MAPTensor(np.loadtxt("memory/feature1/MAP_0.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature1/MAP_1.txt"))]
            elif feat == 2:
                if arch == 0:
                    memory = [torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_0.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_1.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_2.txt"))]
                elif arch == 1:
                    memory = [torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_0.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_1.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_2.txt"))]
                elif arch == 2:
                    memory = [torchhd.BSBCTensor.random(num_chs, n,block_size=5), torchhd.BSBCTensor.random(6, n,block_size=5), torchhd.level(levels, n, "BSBC", block_size=5)]
                shape = 7
                with open("minmax_sp.npy", 'rb') as f:
                    min_sp = np.load(f)
                    max_sp = np.load(f)
            for i in range(len(memory)):
                mem = np.array(memory[i])
                np.savetxt(f"memory/feature{feat}/{archi}_{i}.txt",mem)
            patientn = 0
            for patient in training_patients:
                for result in ["seizures", "non-seizures"]:
                    files_patient = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                    for exc in files_patient:
                        seizure_hv = np.zeros(n)
                        nonseizure_hv = np.zeros(n)
                        files_count = len(files_patient) - 1
                        files_arr = torchhd.identity(files_count,n,archi)
                        outer_counter = 0
                        for file in files_patient:
                            if exc == file:
                                continue
                            else:
                                path = f"features/perfile/feature{feat}/{patient}/{result}/{file}"
                                feat_arr = np.loadtxt(path)
                                if feat == 1:
                                    shape = int(len(feat_arr[0])/17)
                                feat_arr = feat_arr.reshape(-1,shape)
                                print(feat_arr)
                                wins = int(len(feat_arr)/17)
                                tot_arr = torchhd.identity(wins,n,archi)
                                win_arr = torchhd.identity(num_chs,n,archi)
                                counter = 0
                                for window in feat_arr:
                                    
                                    chan = counter%num_chs
                                    wincount = int((counter-counter%num_chs)/num_chs)
                                    feat_len = len(window)
                                    sp_arr = torchhd.identity(6,n,archi)
                                    lbp_arr = torchhd.identity(feat_len-1,n,archi)
                                    perm = 0
                                    for i in range(feat_len):
                                        if i==0:
                                            channel = memory[0][int(window[0])-1]
                                        else:
                                            if feat == 1:
                                                lbp = memory[1][int(window[i])]
                                                res = lbp.permute(perm)
                                                perm+=1
                                                lbp_arr[i-1] = res
                                            elif feat == 2:
                                                sp_id = memory[1][i-1]
                                                sp_min = min_sp[patientn][i-1]
                                                sp_max = max_sp[patientn][i-1]
                                                quantized_value = int((window[i] - sp_min) / ((sp_max - sp_min)  / (levels - 1)))
                                                sp_val = memory[2][quantized_value]
                                                res = sp_id.bind(sp_val)
                                                sp_arr[i-1] = res
                                                
                                    if feat == 1:
                                        lbp_arr_fin = lbp_arr.multibundle()
                                        res = lbp_arr_fin.bind(channel)
                                        lbp_arr = torchhd.identity(feat_len-1,n,archi)
                                        win_arr[chan] = res
                                    elif feat == 2:
                                        sp_arr_fin = sp_arr.multibundle()
                                        res = sp_arr_fin.bind(channel)
                                        sp_arr = torchhd.identity(6,n,archi)
                                        win_arr[chan] = res
                                        
                                    if chan == 16:
                                        
                                        res = win_arr.multibundle().normalize()
                                        if wincount != 0:
                                            prev_win_arr = tot_arr[wincount-1]
                                        else:
                                            prev_win_arr = torchhd.identity(1,n,archi)
                                        tot_arr[wincount] = res
                                        win_arr = torchhd.identity(num_chs,n,archi)
                                        # print(f"win_arr {res}, {torchhd.hamming_similarity(prev_win_arr,res)}")
                                        
                                    counter += 1
                                res = tot_arr.multibundle().normalize()
                                files_arr[outer_counter] = res
                                print(f"tot_arr {res}")
                                print(f"{outer_counter+1} out of {files_count}")
                                outer_counter += 1
                        res = files_arr.multibundle().normalize()
                        outp = np.array(res)
                        filename_out = f"assocmem/{archi}/feature{feat}/{patient}/{result}/{exc}.txt"   
                        np.savetxt(filename_out, outp)
                        print(f"{exc} success")
                patientn += 1
    # """
    # TESTING
    patientn = 1
    testing_patients = ["chb"+str(x).zfill(2) for x in range(patientn, 5+1)]
    architecture = [0]
    testing_features = [1]
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open("hdc_testing_log_3.txt", 'a') as log:
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
        elif arch == 1:
            archi = "MAP"
        elif arch == 2:
            archi = "BSBC"
        with open("hdc_testing_log_3.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Now Testing {archi} Architecture
                ---

                """
            ))
        for feat in testing_features:
            if feat == 1:
                if arch == 0:
                    memory = [torchhd.BSCTensor(np.loadtxt("memory/feature1/BSC_0.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature1/BSC_1.txt"))]
                if arch == 1:
                    memory = [torchhd.MAPTensor(np.loadtxt("memory/feature1/MAP_0.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature1/MAP_1.txt"))]
            elif feat == 2:
                if arch == 0:
                    memory = [torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_0.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_1.txt")),torchhd.BSCTensor(np.loadtxt("memory/feature2/BSC_2.txt"))]
                elif arch == 1:
                    memory = [torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_0.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_1.txt")),torchhd.MAPTensor(np.loadtxt("memory/feature2/MAP_2.txt"))]
                elif arch == 2:
                    continue
                shape = 7
                with open("minmax_sp.npy", 'rb') as f:
                    min_sp = np.load(f)
                    max_sp = np.load(f)
            patientn -= 1
            for patient in testing_patients:
                with open("hdc_testing_log_3.txt", 'a') as log:
                            log.write(dedent(
                                f"""
                                ---
                                {patient}
                                ---"""
                            ))
                for result in ["seizures", "non-seizures"]:
                    files_patient = os.listdir(f"features/perfile/feature{feat}/{patient}/{result}")
                    files_seizures = os.listdir(f"features/perfile/feature{feat}/{patient}/seizures")
                    files_nonseizures = os.listdir(f"features/perfile/feature{feat}/{patient}/non-seizures")
                    files_count = len(files_patient)
                    counter = 0
                    for exc in files_patient:
                        
                        path = f"features/perfile/feature{feat}/{patient}/{result}/{exc}"
                        if arch == 0:
                            am = [torchhd.BSCTensor(np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/seizures/{files_seizures[counter]}.txt")),torchhd.BSCTensor(np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/non-seizures/{files_nonseizures[counter]}.txt"))]
                        elif arch == 1:
                            am = [torchhd.MAPTensor(np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/seizures/{files_seizures[counter]}.txt")),torchhd.MAPTensor(np.loadtxt(f"assocmem/{archi}/feature{feat}/{patient}/non-seizures/{files_nonseizures[counter]}.txt"))]
                        feat_arr = np.loadtxt(path)
                        if feat == 1:
                            shape = int(len(feat_arr[0])/17)
                        feat_arr = feat_arr.reshape(-1,shape)
                        wins = int(len(feat_arr)/17)
                        win_arr = torchhd.identity(num_chs,n,archi)
                        counter2 = 0
                        sens = 0
                        spec = 0
                        seizcount = 0
                        nonseizcount = 0
                        for window in feat_arr:
                            chan = counter2%num_chs
                            wincount = int((counter2-counter2%num_chs)/num_chs)
                            feat_len = len(window)
                            sp_arr = torchhd.identity(6,n,archi)
                            lbp_arr = torchhd.identity(feat_len-1,n,archi)
                            
                            for i in range(feat_len):
                                if i==0:
                                    channel = memory[0][int(window[0])-1]
                                    perm = 0
                                else:
                                    if feat == 1:
                                        lbp = memory[1][int(window[i])]
                                        res = lbp.permute(perm)
                                        perm+=1
                                        lbp_arr[i-1] = res
                                    elif feat == 2:
                                        sp_id = memory[1][i-1]
                                        sp_min = min_sp[patientn][i-1]
                                        sp_max = max_sp[patientn][i-1]
                                        quantized_value = int((window[i] - sp_min) / ((sp_max - sp_min)  / (levels - 1)))
                                        sp_val = memory[2][quantized_value]
                                        res = sp_id.bind(sp_val).permute(i-1)
                                        sp_arr[i-1] = res.bind(sp_arr[i-1])
                                        
                            if feat == 1:
                                lbp_arr_fin = lbp_arr.multibundle()
                                res = lbp_arr_fin.bind(channel)
                                lbp_arr = torchhd.identity(feat_len-1,n,archi)
                                win_arr[chan] = res
                            elif feat == 2:
                                sp_arr = sp_arr.multibundle().normalize()
                                res = sp_arr.bind(channel).normalize()
                                win_arr[chan] = res.bind(win_arr[chan])
                            if chan == 16:
                                res = win_arr.multibundle().normalize()
                                win_arr = torchhd.identity(num_chs,n,archi)
                                if arch == 0:
                                    seiz = torchhd.hamming_similarity(res, am[0])
                                    nonseiz = torchhd.hamming_similarity(res, am[1])
                                    print(f"win {wincount} of {wins}")
                                    if seiz > nonseiz:
                                        seizcount+=1
                                    else:
                                        nonseizcount+=1
                                else:
                                    seiz = torchhd.cosine_similarity(res, am[0])
                                    nonseiz = torchhd.cosine_similarity(res, am[1])
                                    # print(f"win {wincount}")
                                    if seiz > nonseiz:
                                        seizcount+=1
                                    else:
                                        nonseizcount+=1
                            counter2 += 1
                        if result == "seizures":
                            sens = seizcount/wins
                        else:
                            spec = nonseizcount/wins
                        log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                        with open("hdc_testing_log_3.txt", 'a') as log:
                                log.write(dedent(
                                    f"""
                                    {exc[6:-9]}, {seizcount}, {nonseizcount}"""
                                ))
                        print(f"seizure sim: {seiz}, nonseizure sim: {nonseiz}")
                        print(f"{counter+1} out of {files_count}")
                        counter += 1
                patientn+=1
    


                                        
                                            
