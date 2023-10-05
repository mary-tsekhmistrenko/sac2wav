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
import shutil

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

# --- copy process_unit_wav.py to obspy path to make sure it is always up to date
### XXX for future version check first if there is a difference in the file and then copy it over ###
shutil.copyfile('./process_unit_wav.py', os.path.join(inp.path_to_obspy, 'process_unit_wav.py'))

# --- executing the main part
if inp.download:
    command = generate_obspyDMT_command(inp)
    cprint("src_sac2wav.runs2w", "[INFO]", bc.blue, f'Executing following command line: \n\n {command} \n\n')
    subprocess.run(command, shell=True)

# --- extract statiton event information from downloaded folder
sta_ev_df = read_station_information(inp)

# --- combine into one large WAV file
# import ipdb; ipdb.set_trace()
if (inp.mode == 'continuous') & (inp.export_type == 0):
    cprint("run_sac2wav.py", "[INFO]", bc.green, "Creating one (big) multichannel/multiday WAV file from continuous dataset.")
    # multicha_multiday_wav_files(inp, sta_ev_df)
    multicha_multiday_wav_files2(inp, sta_ev_df)
elif ((inp.mode == 'continuous') or (inp.mode == 'day')) & (inp.export_type == 1):
    cprint("run_sac2wav.py", "[INFO]", bc.green, "Creating one multichannel WAV file from continuous dataset per downloaded folder.")
    # multicha_multiday_wav_files(inp, sta_ev_df)
    multicha_multiday_wav_files2(inp, sta_ev_df)
elif (inp.mode == 'event') & (inp.export_type == 1):
    cprint("run_sac2wav.py", "[INFO]", bc.green, "Creating one multichannel WAV file from day/event dataset.")
    multicha_multiday_wav_files2(inp, sta_ev_df)
else:
    cprint("run_sac2wav.py", "[WARNING]", bc.orange, 'This export combination is not implemented. Try another setting please. Forced exit for now.')
    sys.exit()

# --- 
cprint(
    "run_sac2wav.py",
    "[Finished]",
    bc.green,
    f'Elapsed time {time.strftime("%H:%M:%S", time.gmtime(time.time() - tic))}',
)

# --- EOF 
