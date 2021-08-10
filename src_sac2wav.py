#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  src_sac2wav
#   Purpose:   
#   Author:    Mary-Tsekhmistrenko
#   Comments:  
#   Version:   0.1
# -------------------------------------------------------------------

# -----------------------------------------------------------------------
# ----------------Import required Modules (Python and Obspy)-------------
# -----------------------------------------------------------------------

# makes interactive plotting possible
# import matplotlib
# matplotlib.use("Qt5Agg")

import os
import sys
import time
import glob
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from obspy import read, UTCDateTime, Stream, Trace
from soundfile import SoundFile
from natsort import natsorted


# -----------------------------------------------------------------------
# ------------------------------ FUNCTIONS ------------------------------
# -----------------------------------------------------------------------

# for the purpose of this script only following channel groups are considered for exporting WAV files. 
# The following list, which acts as a look up table can be extended to accomondate more channel groups.
target_cha = [
                ["BHE", "BHN", "BHZ", "BDH"],
                ["BHX", "BHY", "BHZ", "BDH"],
                ["BH1", "BH2", "BHZ", "BDH"],

                ["HHE", "HHN", "HHZ", "HDH"],
                ["HHX", "HHY", "HHZ", "HDH"],
                ["HH1", "HH2", "HHZ", "HDH"],

                ["LHE", "LHN", "LHZ", "LDH"],
                ["LHX", "LHY", "LHZ", "LDH"],
                ["LH1", "LH2", "LHZ", "LDH"],

                ["MHE", "MHN", "MHZ", "MDH"],
                ["MHX", "MHY", "MHZ", "MDH"],
                ["MH1", "MH2", "MHZ", "MDH"],

                ["MHE", "MHN", "MHZ", "MDH"],
                ["MHX", "MHY", "MHZ", "MDH"],
                ["MH1", "MH2", "MHZ", "MDH"],

                ["VME", "VMN", "VMZ", "VDH"],
                ["VMX", "VMY", "VMZ", "VDH"],
                ["VM1", "VM2", "VMZ", "VDH"]
            ]

def read_station_information(save_path, main_folder):
   
    # if not restricted in the input then all available stations will be searched
    files = glob.glob(os.path.join(save_path, main_folder, 'info', 'station_event'))
    files.sort()

    sta_list = np.array([])

    for file in files:
        # continous expample of station_event file
        # YV,RR48,00,BDH,-27.5792,65.943,-4830.0,0.0,RESIF,continuous01,-12345,-12345,-12345,-12345,0.0,0.0,10,
        try:
            sta_info = np.loadtxt(file, delimiter=',', dtype=object)
            sta_list = np.append(sta_list, sta_info)
        except Exception as exp:
            # in some cases there might be a gap in data; in that case we just skip that day/event;
            pass

    # YS,CCHM,00,BHE,20.771,-155.996994,60.0,0.0,IRIS,continuous31,-12345,-12345,-12345,-12345,90.0,0.0,10,
    df = pd.DataFrame(data=sta_list.reshape(-1,18), columns=["net", "station", "location", "channel", "stalat", "stalon", "staele", 
                                                            "None", "cata", 
                                                            "mode", "evlat", "evlon", "evdep", "mag", 
                                                            "None", "None", "10", "None"])
    return df


def generate_output_folders(mode, save_path):
    # Create a WAV folder if it does not already exisit
    if mode == 'event':
        wav_save = os.path.join(save_path, 'WAV_events')
        if not os.path.isdir(wav_save):
            os.makedirs(wav_save)

    elif mode == 'continuous':
        wav_save = os.path.join(save_path, f'WAV_continuous')
        if not os.path.isdir(wav_save):
            os.makedirs(wav_save)

    elif mode == 'day':
        wav_save = os.path.join(save_path, 'WAV_day')
        if not os.path.isdir(wav_save):
            os.makedirs(wav_save)
    else:
        sys.exit(f'Mode: {mode} does not exist or is not implemented. Forced exit!')

    # Create folder to save figs
    save_fig_path = os.path.join(save_path, 'plots')
    if not os.path.isdir(save_fig_path):
        os.makedirs(save_fig_path)
    
    return wav_save, save_fig_path

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

def export_continuous(df, poly_wav, dmt_folder, folder_to_process, proc_wavs_continuous, station_selection, channel_selection,
                      framerate, bitrate, norm, wav_save, plot_waveforms):

    fsr = open(os.path.join(wav_save, 'sampling_rate_information.txt'), 'w')
    fsr.write('# station \t|\t channel \t|\t sampling rate \t|\t frame rate \t|\t conversion: 60 min SAC to XX min WAV \n')

    uniq_modes = natsorted(df['mode'].unique())
    uniq_sta = df['station'].unique()

    if folder_to_process in 'processed':
        proc_folder = 'processed'
        fold_2_proc = 'p'
    elif folder_to_process in 'proc_resamp':
         proc_folder = 'proc_resamp'
         fold_2_proc = 'r'
    elif folder_to_process in 'proc_instr':
        proc_folder = 'proc_instr'
        fold_2_proc = 'i'
    elif folder_to_process in 'noinstr_noresamp':
        proc_folder = 'proc_noinstr_noresamp'
        fold_2_proc = 'ninr'
    else:
        sys.exit(f'{folder_tor_process} is not defines. Forced Exit.')

    counter = 0
    days_to_proc = []
    all_folders = natsorted (glob.glob(os.path.join(dmt_folder, f'continuous*')))
    for folder in all_folders:
        pklo = open(os.path.join(folder, 'info', 'event.pkl'), 'rb')
        pkll = pkl.load(pklo)
        cont_date = pkll['datetime']

        if cont_date < UTCDateTime(proc_wavs_continuous[0]):
            print(f'{UTCDateTime(proc_wavs_continuous[0])} < {cont_date}: Continue searching for starttime.')
            continue

        if proc_wavs_continuous[1] == '*':
            pass
        else:
            if counter >= proc_wavs_continuous[1]:
                break
        cont_folder = os.path.basename(folder)
        days_to_proc.append(cont_folder)
        counter += 1

    if proc_wavs_continuous[2] == '*':
        chunk_to_proc = days_to_proc
        print(f'Days to process:\n\t{chunk_to_proc}')
        proc_continuous(df, uniq_sta, station_selection, channel_selection,target_cha, uniq_modes, chunk_to_proc, proc_folder, 
                        framerate, bitrate, norm, poly_wav, fsr, fold_2_proc, dmt_folder, wav_save, plot_waveforms)

    else:
        chunks_to_proc = list(chunks(days_to_proc, proc_wavs_continuous[2] ))
        for chunk_to_proc in chunks_to_proc:
            print(f'Days to process:\n\t{chunk_to_proc}')
            proc_continuous(df, uniq_sta, station_selection, channel_selection,target_cha, uniq_modes, chunk_to_proc, proc_folder, 
                            framerate, bitrate, norm, poly_wav, fsr, fold_2_proc, dmt_folder, wav_save, plot_waveforms)


    fsr.close()


def proc_continuous(df, uniq_sta, station_selection, channel_selection, target_cha, uniq_modes, days_to_proc, proc_folder, 
                    framerate, bitrate, norm, poly_wav, fsr, fold_2_proc, dmt_folder, wav_save, plot_waveforms):

    for i, sta in enumerate(uniq_sta):
        if station_selection != '*' and not sta in station_selection:
            continue
        
        print(f'Working on station: {sta}')
        df_sta = df[df["station"] == sta]

        for loc in df_sta["location"].unique():
            print(f'Location: {loc}')
            df_net = df_sta[df_sta["location"].isin([loc])]
        
            for cha_grp in target_cha:
                
                if cha_grp[0][0] == channel_selection:
                    pass
                elif channel_selection == '*':
                    pass
                else:
                    continue

                df_sta_cha = df_net[df_net["channel"].isin(cha_grp)]

                # because not all channels are available for every for station
                if len(df_sta_cha) == 4*len(uniq_modes) or len(df_sta_cha) == 3*len(uniq_modes):
                    pass
                elif len(df_sta_cha) < 3*len(uniq_modes):
                    continue
                else:
                    continue

                # fill these lists with the traces
                channel_counter = 0
                h_data = None
                e_data = None
                z_data = None
                n_data = None

                print(f'Channel Group: {cha_grp}')
                for j, cha in enumerate(cha_grp):
                    print(f'\n\tChannel {cha}.')
            
                    # rest the trace/stream for every channel
                    st = None
                    
                    l = 0 
                    for k, mod in enumerate(uniq_modes):
        
                        if mod not in days_to_proc:
                            continue
                        else:
                            pass
                        
                        print(f'\t\tFolder {mod}')
                        # import ipdb; ipdb.set_trace()
                        try:
                            chans = glob.glob(os.path.join(dmt_folder, mod, f'{proc_folder}', f'*{sta}*{loc}*{cha}*'))[0]
                            print(f'\t\t\tAdding {os.path.basename(chans)}.')
                        
                        except Exception as exp:
                            print(f'\t\t\t--Missing: {mod}/{proc_folder}/*{sta}*{loc}*{cha}')
                            continue
                        
                        # merge to one long trace
                        if l == 0:
                            st = read(chans)
                        if l > 0:
                            st += read(chans)
                        l += 1
                        st.merge(method=1)
                    
                    if not st:
                        # basically skip the channel
                        continue
                    
                    tr = st[0]
                    print(f'\tLength of trace: {tr.stats.npts}')

                    # data = tr.data / abs(tr.data).max()
                    # data = tr.data
                    
                    # order of channels to export needs to be like this: 
                    # ['*HH', '*HY', '*HZ', '*HX']
                    
                    if tr.stats.channel[-1] in ['E', 'X', '1']:
                        # e_data = data*0.95
                        e_cha = tr.stats.channel
                        e_tr = tr

                        fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                        channel_counter += 1

                    elif tr.stats.channel[-1] in ['N', 'Y', '2']:
                        #n_data = data*0.95
                        n_cha = tr.stats.channel
                        n_tr = tr

                        fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                        channel_counter += 1

                    elif tr.stats.channel[-1] in ['Z']:
                        #z_data = data*0.95
                        z_cha = tr.stats.channel
                        z_tr = tr

                        fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                        channel_counter += 1

                    elif tr.stats.channel[-1] in ['H']:
                        #h_data = data*0.95
                        h_cha = tr.stats.channel
                        h_tr = tr

                        fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                        channel_counter += 1

                    else:
                        sys.exit('This channel does not exist you are loading. Check what is happening! Forced Exit.')

                    fsr.write('\n')
                    # >>> END CHANNEL LOOP <<<<  

                if (channel_counter == 3 and h_data == None):
                    print('\t\t\tAdding a zero trace for the hydrophone channel...')
                    h_data = z_tr.stats.npts*[0]
                    h_tr = Trace(np.array(h_data))
                    h_tr.stats.starttime = z_tr.stats.starttime
                    h_tr.stats.sampling_rate = z_tr.stats.sampling_rate
                    h_tr.stats.channel = '0DH'
                    
                    channel_counter += 1
                    h_cha = '0DH'
                    cha_output = f'3{z_tr.stats.channel[0]}'

                elif channel_counter < 3:
                    print(f'\t\t\t{sta} does not have all channels {cha}. Continue to next station without writing a WAV file.')
                    continue 

                elif channel_counter == 4:
                    cha_output = f'4{z_tr.stats.channel[0]}'
                    pass

                elif (channel_counter == 3 and h_data != None):
                    print('\t\t\t Another channel is missing. Exporting this trace regardless.')
                    pass

                else:
                    print(channel_counter, np.mean(collect_tr[:,0]))
                    sys.exit(f'\nNumber of channels is {channel_counter}. \n\n Something unexpected happened and hence forced exit.')

                fsr.write('\n')
                
                # writing the wave file
                # order of channels to export needs to be like this: 
                # ['*HH', '*HY', '*HZ', '*HX']
                        # writing the wave file

                st = length_checker(Stream(traces=[h_tr, n_tr, e_tr, z_tr]))

                h_da = st.select(channel=h_cha)
                z_da = st.select(channel=z_cha)
                e_da = st.select(channel=e_cha)
                n_da = st.select(channel=n_cha)
                
                print(f'\tNormalizing after merging continuous waveforms.')
                print(f'\tMax values of channels HZEN (might be useful for norm factor):', abs(h_da[0].data).max(), abs(z_da[0].data).max(), abs(e_da[0].data).max(), abs(n_da[0].data).max())

                if norm < 0:
                    if h_da[0].stats.channel == '0DH':
                        h_data = h_da[0].data
                    else:
                        h_data = (h_da[0].data / abs(h_da[0].data).max())*0.95
                    
                    norming = max([abs(z_da[0].data).max(), abs(e_da[0].data).max(), abs(n_da[0].data).max()])

                    z_data = (z_da[0].data / norming)*0.95
                    e_data = (e_da[0].data / norming)*0.95
                    n_data = (n_da[0].data / norming)*0.95
                    print(f'\tNorm factor used for three components: {norming}')

                else:
                    if h_da[0].stats.channel == '0DH':
                        h_data = h_da[0].data
                    else:
                        h_data = (h_da[0].data / norm)
                    
                    z_data = (z_da[0].data / norm)
                    e_data = (e_da[0].data / norm)
                    n_data = (n_da[0].data / norm)
                    print(f'\tNorm factor used for all components: {norm}')

                
                try:
                    collect_tr = np.c_[h_data, n_data, e_data, z_data]
                    collect_cha = [h_cha, n_cha, e_cha, z_cha]
                except Exception as exp:
                    print(f'\n\nError: {exp}\nFor station:{sta}')
                    continue

                if plot_waveforms:
                    plot_waves(sta, collect_tr, collect_cha, tr.stats.sampling_rate, tr.stats.network, tr.stats.location, 'continuous', proc_folder, wav_save)

                if poly_wav:
                    file_name = (f'{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{cha_output}_{fold_2_proc}_{st[0].stats.starttime.date.year}-{st[0].stats.starttime.date.month:02d}-'
                                f'{st[0].stats.starttime.date.day:02d}_{st[0].stats.endtime.date.year}-'
                                f'{st[0].stats.endtime.date.month:02d}-{st[0].stats.endtime.date.day:02d}.WAV')

                    path_file_wav = os.path.join(wav_save, file_name)
                    with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=4, 
                                    subtype=bitrate, endian=None, format=None, closefd=True) as f:
                        f.write(collect_tr)
                    f.close()

                else:
                    for j, cha in enumerate(collect_cha):
                        path_file_wav = os.path.join(wav_save, "%s_%s_%s.WAV" % (tr.stats.station, cha, tr.stats.starttime))
                        with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=1, 
                                    subtype=bitrate, endian=None, format=None, closefd=True) as f:
                            f.write(collect_tr[:,j])
                    f.close() 
                
                # clear memory after the channels have been written
                print ('Clearing memeroy...')
                del collect_tr, h_da, z_da, e_da, n_da


# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

def export_day(df, poly_wav, dmt_folder, folder_to_process, proc_wavs_days, station_selection, channel_selection,
                      framerate, bitrate, wav_save, plot_waveforms):

    # save this in the WAV folder
    fsr = open(os.path.join(wav_save, 'sampling_rate_information.txt'), 'w')
    fsr.write('# station \t|\t channel \t|\t sampling rate \t|\t frame rate \t|\t conversion: 60 min SAC to XX min WAV \n')

    uniq_modes = natsorted(df['mode'].unique())
    uniq_sta = df['station'].unique()

    counter = 0
    days_to_proc = []
    all_folders = natsorted (glob.glob(os.path.join(dmt_folder, f'continuous*')))
    
    for folder in all_folders:
        pklo = open(os.path.join(folder, 'info', 'event.pkl'), 'rb')
        pkll = pkl.load(pklo)
        cont_date = pkll['datetime']
        if cont_date < UTCDateTime(proc_wavs_days[0]):
            print(f'{UTCDateTime(proc_wavs_days[0])} < {cont_date}: Continue searching for starttime.')
            continue

        if counter >= proc_wavs_days[1]:
            break

        cont_folder = os.path.basename(folder)
        days_to_proc.append(cont_folder)
        counter += 1
    
    print(f'Days to process:\n{days_to_proc}')

    if folder_to_process in 'processed':
        proc_folder = 'processed'
    elif folder_to_process in 'proc_resamp':
         proc_folder = 'proc_resamp'
    elif folder_to_process in 'proc_instr':
        proc_folder = 'proc_instr'
    elif folder_to_process in 'noinstr_noresamp':
        proc_folder = 'proc_noinstr_noresamp'
    else:
        sys.exit(f'{folder_tor_process} is not defines. Forced Exit.')

    for mod in uniq_modes:
        if mod not in days_to_proc:
            continue
        else:
            pass

        print(f'Searching SACs in {mod}')

        for loc in df["location"].unique():
            print(f'\tLocation: {loc}')
            df_net = df[df["location"].isin([loc])]

            for sta in uniq_sta:
                if station_selection != '*' and not sta in station_selection:
                    continue

                print (f'\tWorking on station: {sta}')
                df_sta = df_net[df_net["station"] == sta]
                df_sta_mod = df_sta[df_sta["mode"] == mod]

                for cha_grp in target_cha:
                    if cha_grp[0][0] == channel_selection:
                        pass
                    elif channel_selection == '*':
                        pass
                    else:
                        continue
                    # since it is possible to have different channel groups for one station this part
                    # ensures that 
                    df_grp = df_sta_mod[df_sta_mod["channel"].isin(cha_grp)]

                    if len(df_grp) == 4 or len(df_grp) == 3:
                        pass
                    elif len(df_grp) < 3:
                        continue
                    else:
                        continue

                    print(f'\t\tSearching in {cha_grp}')

                    # fill these lists with the traces
                    channel_counter = 0
                    h_data = None
                    e_data = None
                    z_data = None
                    n_data = None

                    for cha in cha_grp:

                        try:
                            chan = glob.glob(os.path.join(dmt_folder, mod, proc_folder, f'*{sta}*{loc}*{cha}*'))[0]
                            print (f'\t\t\tAdding {os.path.basename(chan)}')
                            tr = read(chan)[0]
                            # data = tr.data / abs(tr.data).max()

                        except Exception as exp:
                            # print(f'{exp} \nNo station available in:')
                            # print(os.path.join(save_path,  main_folder, proc_folder, f'*{sta}*{cha}*'))
                            print(f'\t\t\t--Missing: {mod}/{proc_folder}/*{sta}*{loc}*{cha}')
                            continue

                        # order of channels to export needs to be like this: ['HHH', 'HHY', 'HHZ', 'HHX']
                        if tr.stats.channel[-1] in ['E', 'X', '1']:
                            # e_data = data*0.95
                            e_cha = tr.stats.channel
                            e_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['N', 'Y', '2']:
                            # n_data = data*0.95
                            n_cha = tr.stats.channel
                            n_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['Z']:
                            # z_data = data*0.95
                            z_cha = tr.stats.channel
                            z_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['H']:
                            # h_data = data*0.95
                            h_cha = tr.stats.channel
                            h_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        else:
                            sys.exit('\nThis channel does not exist you are loading. Check what is happening!')

                    if (channel_counter == 3 and h_data == None):
                        print('\t\t\tAdding a zero trace for the hydrophone channel...')
                        h_data = z_tr.stats.npts*[0]
                        h_tr = Trace(np.array(h_data))
                        h_tr.stats.starttime = z_tr.stats.starttime
                        h_tr.stats.sampling_rate = z_tr.stats.sampling_rate
                        h_tr.stats.channel = '0DH'
                        
                        channel_counter += 1
                        h_cha = '0DH'

                    elif channel_counter < 3:
                        print(f'\t\t\t{sta} does not have all channels (more than 2 channels are missing). \n\t\t\tContinue to next station without writing a WAV file.')
                        continue 

                    elif channel_counter == 4:
                        pass

                    elif (channel_counter == 3 and h_data != None):
                        print('\t\t\t Another channel is missing. Exporting this trace regardless.')
                        pass

                    else:
                        print(channel_counter, np.mean(collect_tr[:,0]))
                        sys.exit(f'\nNumber of channels is {channel_counter}. \n\n Something unexpected happened and hence forced exit.')

                    fsr.write('\n')

                    st = length_checker(Stream(traces=[h_tr, n_tr, e_tr, z_tr]))
                    
                    h_da = st.select(channel=h_cha)
                    z_da = st.select(channel=z_cha)
                    e_da = st.select(channel=e_cha)
                    n_da = st.select(channel=n_cha)

                    if h_da[0].stats.channel == '0DH':
                        h_data = h_da[0].data
                    else:
                        h_data = (h_da[0].data / abs(h_da[0].data).max())*0.95
                    z_data = (z_da[0].data / abs(z_da[0].data).max())*0.95
                    e_data = (e_da[0].data / abs(e_da[0].data).max())*0.95
                    n_data = (n_da[0].data / abs(n_da[0].data).max())*0.95

                    print(f'\tNormalizing channelwise after merging continuous waveforms.')
                    try:
                        collect_tr = np.c_[h_data, n_data, e_data, z_data]
                        collect_cha = [h_cha, n_cha, e_cha, z_cha]
                    except Exception as exp:
                        print(f'\n\nError: {exp}\nFor station:{sta}')
                        continue

                    try:
                        collect_tr = np.c_[h_data, n_data, e_data, z_data]
                        collect_cha = [h_cha, n_cha, e_cha, z_cha]
                    except Exception as exp:
                        print(f'\n\nError: {exp}\nFor station:{sta}')
                        continue

                    if plot_waveforms:
                        plot_waves(sta, collect_tr, collect_cha, tr.stats.sampling_rate, tr.stats.network, tr.stats.location, f'day_{mod}', proc_folder, wav_save)

                    if poly_wav:
                        file_name = (f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}_{collect_cha[0]}_{collect_cha[1]}_{collect_cha[2]}_{collect_cha[3]}_{folder_to_process}_{tr.stats.starttime.date.year}-{tr.stats.starttime.date.month:02d}-'
                                    f'{tr.stats.starttime.date.day:02d}T{tr.stats.starttime.time.hour:02d}-'
                                    f'{tr.stats.starttime.time.minute:02d}-{tr.stats.starttime.time.second:02d}.WAV')

                        path_file_wav = os.path.join(wav_save, file_name)
                        with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=4, 
                                        subtype=bitrate, endian=None, format=None, closefd=True) as f:
                            f.write(collect_tr)
                        f.close()

                    else:
                        for j, cha in enumerate(collect_cha):

                            path_file_wav = os.path.join(wav_save, "%s_%s_%s_%s_%s.WAV" % (tr.stats.station, cha, mod, proc_folder, tr.stats.starttime))
                            with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=1, 
                                        subtype=bitrate, endian=None, format=None, closefd=True) as f:
                                f.write(collect_tr[:,j])

                            f.close()
    # clear memory
    del collect_tr, h_da, z_da, e_da, n_da
    fsr.close()

# -----------------------------------------------------------------------
#
# -----------------------------------------------------------------------

def export_event(df, poly_wav, dmt_folder, folder_to_process, 
                 proc_wavs_events, station_selection, channel_selection, 
                 framerate, bitrate, wav_save, plot_waveforms):
    

    # save this in the WAV folder
    fsr = open(os.path.join(wav_save, 'sampling_rate_information.txt'), 'w')
    fsr.write('# station \t|\t channel \t|\t sampling rate \t|\t frame rate \t|\t conversion: 60 min SAC to XX min WAV \n')

    uniq_modes = df['mode'].unique()
    uniq_sta = df['station'].unique()

    if folder_to_process in 'processed':
        proc_folder = 'processed'
    elif folder_to_process in 'proc_resamp':
         proc_folder = 'proc_resamp'
    elif folder_to_process in 'proc_instr':
        proc_folder = 'proc_instr'
    elif folder_to_process in 'noinstr_noresamp':
        proc_folder = 'proc_noinstr_noresamp'
    else:
        sys.exit(f'{folder_tor_process} is not defines. Forced Exit.')
    
    for mod in uniq_modes:
        
        # only selected events
        if mod in proc_wavs_events:
            pass
        # let everything pass
        elif '*_*.*' in proc_wavs_events:
            pass
        # skip
        else:
            continue

        print(f'Searching SACs in {mod}')

        for loc in df["location"].unique():
            print(f'\tLocation: {loc}')
            df_net = df[df["location"].isin([loc])]

            for sta in uniq_sta:
                if station_selection != '*' and not sta in station_selection:
                    continue

                print (f'\tReading for {sta}')
                df_sta = df_net[df_net["station"] == sta]
                df_sta_mod = df_sta[df_sta["mode"] == mod]

                for cha_grp in target_cha:
                    if cha_grp[0][0] == channel_selection:
                        pass
                    elif channel_selection == '*':
                        pass
                    else:
                        continue
                    # since it is possible to have different channel groups for one station this part
                    # ensures that 
                    df_grp = df_sta_mod[df_sta_mod["channel"].isin(cha_grp)]

                    if len(df_grp) == 4 or len(df_grp) == 3:
                        pass
                    elif len(df_grp) < 3:
                        continue
                    else:
                        continue
        
                    print(f'\t\tSearching in {cha_grp}')
            
                    # fill these lists with the traces
                    channel_counter = 0
                    h_data = None
                    e_data = None
                    z_data = None
                    n_data = None

                    for cha in cha_grp:
                        try:
                            
                            chan = glob.glob(os.path.join(dmt_folder, mod, proc_folder, f'*{sta}*{loc}*{cha}*'))[0]
                            print (f'\t\t\tAdding {os.path.basename(chan)}')
                            tr = read(chan)[0]
                            # data = tr.data / abs(tr.data).max()

                        except Exception as exp:
                            print(f'\t\t\t--Missing: {mod}/{proc_folder}/*{sta}*{loc}*{cha}')
                            continue

                        # order of channels to export needs to be like this: ['HHH', 'HHY', 'HHZ', 'HHX']
                        if tr.stats.channel[-1] in ['E', 'X', '1']:
                            # e_data = data*0.95
                            e_cha = tr.stats.channel
                            e_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['N', 'Y', '2']:
                            # n_data = data*0.95
                            n_cha = tr.stats.channel
                            n_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['Z']:
                            # z_data = data*0.95
                            z_cha = tr.stats.channel
                            z_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        elif tr.stats.channel[-1] in ['H']:
                            # h_data = data*0.95
                            h_cha = tr.stats.channel
                            h_tr = tr

                            fsr.write(f'{sta} \t\t {tr.stats.channel} \t\t {tr.stats.sampling_rate} \t\t {framerate} \t\t {(60*60*tr.stats.sampling_rate)/framerate} \n')
                            channel_counter += 1

                        else:
                            sys.exit('\nThis channel does not exist you are loading. Check what is happening!')

                    if (channel_counter == 3 and h_data == None):
                        print('\t\t\tAdding a zero trace for the hydrophone channel...')
                        h_data = z_tr.stats.npts*[0]
                        h_tr = Trace(np.array(h_data))
                        h_tr.stats.starttime = z_tr.stats.starttime
                        h_tr.stats.sampling_rate = z_tr.stats.sampling_rate
                        h_tr.stats.channel = '0DH'
                        
                        channel_counter += 1
                        h_cha = '0DH'

                    elif channel_counter < 3:
                        print(f'\t\t\t{sta} does not have all channels {cha}. Continue to next station without writing a WAV file.')
                        continue 

                    elif channel_counter == 4:
                        pass

                    elif (channel_counter == 3 and h_data != None):
                        print('\t\t\t Another channel is missing. Exporting this trace regardless.')
                        pass

                    else:
                        print(channel_counter, np.mean(collect_tr[:,0]))
                        sys.exit(f'\nNumber of channels is {channel_counter}. \n\n Something unexpected happened and hence forced exit.')

                    fsr.write('\n')

                    # writing the wave file
                    st = length_checker(Stream(traces=[h_tr, n_tr, e_tr, z_tr]))
                    
                    h_da = st.select(channel=h_cha)
                    z_da = st.select(channel=z_cha)
                    e_da = st.select(channel=e_cha)
                    n_da = st.select(channel=n_cha)

                    if h_da[0].stats.channel == '0DH':
                        h_data = h_da[0].data
                    else:
                        h_data = (h_da[0].data / abs(h_da[0].data).max())*0.95
                    z_data = (z_da[0].data / abs(z_da[0].data).max())*0.95
                    e_data = (e_da[0].data / abs(e_da[0].data).max())*0.95
                    n_data = (n_da[0].data / abs(n_da[0].data).max())*0.95

                    print(f'\tNormalizing channelwise after merging continuous waveforms.')

                    try:
                        collect_tr = np.c_[h_data, n_data, e_data, z_data]
                        collect_cha = [h_cha, n_cha, e_cha, z_cha]
                    except Exception as exp:
                        print(f'\n\nError: {exp}\nFor station:{sta}')
                        continue
                
                    if plot_waveforms:
                        date_name = (f'{tr.stats.starttime.date.year}-{tr.stats.starttime.date.month:02d}-'
                            f'{tr.stats.starttime.date.day:02d}T{tr.stats.starttime.time.hour:02d}-'
                            f'{tr.stats.starttime.time.minute:02d}-{tr.stats.starttime.time.second:02d}')
                        plot_waves(sta, collect_tr, collect_cha, tr.stats.sampling_rate,  tr.stats.network, tr.stats.location, date_name, proc_folder, wav_save)

                    if poly_wav:
                        file_name = (f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}_{collect_cha[0]}_{collect_cha[1]}_{collect_cha[2]}_{collect_cha[3]}_{folder_to_process}_{tr.stats.starttime.date.year}-{tr.stats.starttime.date.month:02d}-'
                                        f'{tr.stats.starttime.date.day:02d}T{tr.stats.starttime.time.hour:02d}-'
                                        f'{tr.stats.starttime.time.minute:02d}-{tr.stats.starttime.time.second:02d}.WAV')
                        path_file_wav = os.path.join(wav_save, file_name)
                        with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=4, 
                                        subtype=bitrate, endian=None, format=None, closefd=True) as f:
                            f.write(collect_tr)
                        f.close()

                    else:
                        for j, cha in enumerate(collect_cha):
                            file_name = (f'{tr.id}_{folder_to_process}_{tr.stats.starttime.date.year}-{tr.stats.starttime.date.month:02d}-'
                                        f'{tr.stats.starttime.date.day:02d}T{tr.stats.starttime.time.hour:02d}-'
                                        f'{tr.stats.starttime.time.minute:02d}-{tr.stats.starttime.time.second:02d}.WAV')
                            path_file_wav = os.path.join(wav_save, file_name)
                            with SoundFile(path_file_wav, 'w', samplerate=framerate, channels=1, 
                                        subtype=bitrate, endian=None, format=None, closefd=True) as f:
                                f.write(collect_tr[:,j])

                            f.close()
    # clear memory
    del collect_tr, h_da, z_da, e_da, n_da
    fsr.close()

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

def plot_waves(sta, collect_tr, collect_cha, sampling_rate, network, location, date_name, proc_folder, wav_save):

    plt.ioff()
    fig, axs = plt.subplots(4,2, figsize=(15, 15), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    fig.subplots_adjust(hspace = 0.5, wspace=0.5)

    j = 0
    for i, cha in enumerate(collect_cha):

        # data = collect_tr[:,i] / abs(collect_tr[:,i]).max()
        data = collect_tr[:,i]
        
        axs[j].plot(data)
        axs[j].set_title(cha, weight='bold')
        axs[j].set_xlabel('samples')
        axs[j].set_ylabel('normalized')
        
        axs[j+1].magnitude_spectrum(data, Fs=sampling_rate, color='C2', alpha=0.5)
        axs[j+1].set_title(f'Spectrum: {cha}', weight='bold')
        axs[j+1].set_xscale('log')
        axs[j+1].set_xlim(0.001, 30)
        j += 2

    plt.savefig(os.path.join(wav_save, f'{network}.{sta}.{location}_{collect_cha[0]}_{collect_cha[1]}_{collect_cha[2]}_{collect_cha[3]}_{proc_folder}_{date_name}.png'), dpi=300)
    plt.clf()
    plt.close()


# -----------------------------------------------------------------------
# check for same length in traces
# -----------------------------------------------------------------------

def length_checker(st):
    
    samples = []
    starttime = []
    endtime = []

    for tr in st:
        samples.append(tr.stats.npts)
        starttime.append(tr.stats.starttime)
        endtime.append(tr.stats.endtime)
    
    if len(np.unique(samples)) > 1:
        st.trim()
        return st
    else:
        return st




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# -----------------------------------------------------------------------
# --------------------------------- END ---------------------------------
# -----------------------------------------------------------------------
