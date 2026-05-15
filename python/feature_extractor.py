import os, time, re
import numpy as np
import hdc_methods as hdc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from textwrap import dedent
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from datetime import datetime
import scipy

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
    window_size = 256 # 1.0 seconds
    window_step = 128 # 0.5 seconds
    fs = 256
    d = 6 # LBP bit size
    num_chs = 17 # constant
    patients_to_extract = ["chb"+str(x).zfill(2) for x in range(13, 15+1)]
    training_features = [1]
    log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open("feature_extraction_log.txt", 'a') as log:
            log.write(dedent(
                f"""
                ---
                Extraction Begin {log_time}
                ---
                #1 - LBP
                #2 - Spectral Power
                #3 - LBP + LL + MA
                #4 - SP + LL

                """
            ))
    for feature in training_features:
        
        for patient in patients_to_extract:
            
            label_exc = "non-seizures"
            for label in ["seizures", "non-seizures"]:
                path_patient = f"chbmit-eeg-processed/{label}/" + patient
                files_patient = os.listdir(path_patient)
                files = len(files_patient)
                for file_inc in files_patient:
                    print(f"{file_inc}")
                    feature_array = np.array([])
                    label_array = np.array([])
                    name_chs, value_chs, last_samp, samp_freq = extract_npy(str(path_patient + "/"+ file_inc))
                    samples = len(value_chs[0])
                    s = range(0, samples, window_step)
                    for j in s:
                        
                        channel = 0
                        # line length and mean amplitude
                        # i = scipy.signal.filtfilt(b_filt, a_filt, i)
                        # i = scipy.signal.upfirdn(h_filt, i)
                        # print(f"Working... {counter} of {samples} - {file_inc}")
                        
                        # print(f"{j} out of {samples}")
                        
                        for m in value_chs:
                            feat_temp = np.array([])
                            channel += 1
                            
                            
                            i = m[j:j+window_size+6]
                            if j+window_size+6 > samples:
                                break
                            feat_temp = np.append(feat_temp, channel)
                            if feature == 1:
                                lbp = hdc.compute_optimizedLBP(i,6)
                                
                                feat_temp = np.append(feat_temp, lbp,0)
                                x = len(feat_temp)
                                # print(f"{x} vs {j} to {j+window_size} out of {samples} vs {feat_temp[0]}")
                                # print(x)
                                # print(feat_temp)
                            elif feature == 2:
                                sp = np.array(hdc.compute_bandPower(i, fs, window_size))
                                feat_temp = np.append(feat_temp, sp)
                            elif feature == 3:
                                lbp = np.histogram(hdc.compute_optimizedLBP(i,6),bins=np.array(range(0,65)))[0]
                                feat_temp = np.append(feat_temp, lbp)
                                ll_ma = np.array([hdc.compute_lineLength(i),hdc.compute_meanAmp(i)])
                                feat_temp = np.append(feat_temp, ll_ma)
                            elif feature == 4:
                                sp = np.array(hdc.compute_bandPower(i, fs, window_size))
                                feat_temp = np.append(feat_temp, sp)
                                ll = np.array([hdc.compute_lineLength(i)])
                                feat_temp = np.append(feat_temp, ll)
                            if len(feat_temp) == 0:
                                continue
                            else:
                                feature_array = np.append(feature_array, feat_temp,0)
                                
                                if label == "non-seizures":
                                    label_array = np.append(label_array, [-1])
                                    
                                else:
                                    label_array = np.append(label_array, [1])
                            # print(f"feat array - {len(feature_array)/(64*17)}, label_array - {len(label_array)}, loop - {counter}")
                    # print(f"file {counter} out of {len(files_patient)}")
            
                    if feature == 1:
                        feature_array = feature_array.reshape(-1,(window_size+1)*17)
                        # print(f"{x} and 17*x = {x*17}")
                    if feature == 2:
                        feature_array = feature_array.reshape(-1,7*17)
                    if feature == 3:
                        feature_array = feature_array.reshape(-1,67*17)
                    if feature == 4:
                        feature_array = feature_array.reshape(-1,8*17)
                    print(len(feature_array))
                    print(len(label_array))
                    print(feature_array)
                    filename_feat = f"features/perfile/feature{feature}/{patient}_{window_size}/{label}/{file_inc[:-4]}_feat.txt"
                    # filename_label = f"features/perfile/feature{feature}/{file_inc[:-4]}_label.txt"
                    np.savetxt(filename_feat, feature_array)
                    # np.savetxt(filename_label, label_array)
            log_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            with open("feature_extraction_log.txt", 'a') as log:
                    log.write(dedent(
                        f"""
                        ---
                        Finished {patient} for feature {feature} at {log_time}

                        """
                    ))


            