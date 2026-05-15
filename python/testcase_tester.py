import os, time, re
import numpy as np
import hdc_methods as hdc
from textwrap import dedent
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import scipy.signal
import random

if __name__ == "__main__":
    
    hv_arr = hdc.generate_randommemory(10000, 0.5, 3)
    for hv in hv_arr:
        hv_str = "".join(map(str,hv))
        hv_str = hex(int(hv_str,2))[2:]
        print(f"{hv_str[:10]}...")
        with open("bundle_tc.txt", 'a') as log:
            log.write(dedent(
                f"""
                {hv_str}
                """
            ))
    print("bundles into")
    hv_out = hdc.bundle(hv_arr)
    hv_out = "".join(map(str,hv_out))
    hv_out_hex = hex(int(hv_out,2))[2:]
    with open("bundle_tc.txt", 'a') as log:
        log.write(dedent(
            f"""
            {hv_out_hex}
            """
        ))
    print(f"{hv_out_hex[:10]}...")
    """
    hv_out = hdc.bind(hv_1, hv_2)
    hv_out = "".join(map(str,hv_out))
    hv_out_hex = hex(int(hv_out,2))
    print(f"HV_1: {hex(hv_1_num)[2:20]}...")
    print(f"HV_2: {hex(hv_2_num)[2:20]}...")
    print("bind to")
    print(f"HV_out: {hv_out_hex[2:20]}...")
    """
