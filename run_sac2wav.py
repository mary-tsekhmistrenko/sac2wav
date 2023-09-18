#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algorithm runs obspyDMT to download sac/mseed data and converts it automatically to WAV files.
Use input.ini to deinfe your desired data (e.g. earthquakes, or continuous day files)

How to run: 
python run_sac2wav.py input.ini
"""

import os
import sys
import time

import subprocess

from src_sac2wav import cprint, bc, InpS2W
from src_sac2wav import *

# ---
tic = time.time()

# --- reading input from command line
try:
    inp_file = os.path.abspath(sys.argv[1])
    inp = InpS2W(inp_file=inp_file)
except Exception as exp1:
    cprint("run_sac2wav.py", "[ERROR]", bc.red, exp1)
    cprint("run_sac2wav.py", "[ERROR]", bc.red,
           "Did you run the code like this: python run_sac2wav.py <path/to/input> ?")
    sys.exit("Forced exit while attempting to read input file.")

# --- executing the main part
if inp.download:
    command = generate_obspyDMT_command(inp)
    cprint("src_sac2wav.runs2w", "[INFO]", bc.blue, f'Executing following command line: \n\n {command} \n\n')
    subprocess.run(command, shell=True)

# --- extract statiton event information from downloaded folder
sta_ev_df = read_station_information(inp)

# --- combine into one large WAV file
if inp.mode == 'continuous':
    cprint("run_OBSToolBox.py", "[INFO]", bc.green, "Creating one multichannel/multiday WAV file from continuous dataset.")
    multicha_multiday_wav_files(inp, sta_ev_df)
elif (inp.mode == 'event') or (inp.mode == 'day'):
    cprint("run_OBSToolBox.py", "[INFO]", bc.green, "Creating one multichannel WAV file from day/event dataset.")
    multicha_wav_files(inp, sta_ev_df)

# --- 
cprint(
    "run_OBSToolBox.py",
    "[Finished]",
    bc.green,
    f'Elapsed time {time.strftime("%H:%M:%S", time.gmtime(time.time() - tic))}',
)

# --- EOF 
