import numpy as np
from tqdm import tqdm

# Q12.3 signed fixed rep!! from uV values
num_int = 13 # including sign
num_frac = 3
patient_file = "chb01_03_0"
with open(f"chbmit-eeg-processed/seizures/chb01/{patient_file}.npy", 'rb') as f:
    label_chs = np.load(f)
    value_chs = np.load(f)
    last_samp = np.load(f)
    samp_freq = np.load(f)
    written_string = ""

    # num_values = 6  + 4 + (2*2) # LBP size + window size + window step * number of windows
    num_values = len(value_chs[0])
    num_channels = len(value_chs)
    for i in tqdm(range(num_values)):
        written_string += "en = 1;\n"
        for j in range(num_channels):
            value = value_chs[j][i] * (10**6)
          
            scaled_value = value * (2**num_frac)
            bin_value = bin(abs(round(scaled_value)))

            if value >= 0:
                if len(bin_value) > num_int + num_frac + 2:
                    raise ValueError("Value too large for Q13.3 fixed rep")
                bin_value = bin_value.removeprefix("0b").rjust(num_int + num_frac, "0")
            else:
                if len(bin_value) > num_int + num_frac + 3:
                    raise ValueError("Value too large for Q13.3 fixed rep")
                bin_value = list(bin_value.removeprefix("0b").rjust(num_int + num_frac, "0"))
                for k in range(len(bin_value)):
                    if bin_value[k] == "0":
                        bin_value[k] = "1"
                    else:
                        bin_value[k] = "0"
                bin_value = "".join(bin_value)
                bin_value = bin(int(bin_value, 2) + 1)
                bin_value = bin_value.removeprefix("0b").rjust(num_int, "1")

            written_string += f"samples[{j}] = " + f"{num_int + num_frac}'b{bin_value}" + ";\n"
        written_string += "# (500 + 500)\nen = 0;\n# (3906250 - 500 - 500)\n\n"

with open(f"tb_encoder_sv_{patient_file}_s.txt", "w") as f:
    f.write(written_string) 