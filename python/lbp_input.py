import numpy as np
from tqdm import tqdm

# Q12.3 signed fixed rep!! from uV values
num_int = 6
patient_file = "chb01_03_0"

written_string = ""
value_chs = np.loadtxt("lbp_window1/lbp_full_first10.txt")
# num_values = 6  + 4 + (2*2) # LBP size + window size + window step * number of windows
num_values = len(value_chs)
num_channels = len(value_chs[0])
for i in tqdm(range(num_values)):
    written_string += "en = 1;\n"
    for j in range(num_channels):
        scaled_value = value_chs[i][j]
        bin_value = bin(abs(round(scaled_value)))

        
        if len(bin_value) > num_int + 2:
            raise ValueError("Value too large for Q13.3 fixed rep")
        bin_value = bin_value.removeprefix("0b").rjust(num_int, "0")

        written_string += f"samples[{j}] = " + f"{num_int}'b{bin_value}" + ";\n"
    written_string += "# (500 + 500)\nen = 0;\n# (3906250 - 500 - 500)\n\n"

with open(f"tb_lbp_input_{patient_file}_s.txt", "w") as f:
    f.write(written_string) 