# ------------- required Python and obspy modules are imported in this part
from obspy.core import read
import os
import errno

from .utils.instrument_handler import instrument_correction
from .utils.resample_handler import resample_unit
from .utils.utility_codes import convert_to_sac

from soundfile import SoundFile

# -----------------------------------------------------------------------------

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ===================== YOU CAN CHANGE THE FOLLOWING FUNCTION =================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# * IMPORTANT *
# The following function (process_unit) is in the waveform level.
# This means that you can write your process unit for one trace, and
# obspyDMT uses this function to pre-process all your waveforms,
# either right after retrieval or as a separate step:
# obspyDMT --datapath /your/dataset --local

# ========== process_unit has the following arguments:
# Use the following parameters to write your process_unit:
# 1. tr_add: address of one trace in your dataset. You can use that to
# read in the data.
# 2. target_path: address of the event that should be processed.
# 3. input_dics: dictionary that contains all the inputs.
# 4. staev_ar: an array that contains the following information:
# net, sta, loc, cha, station latitude, station longitude, station elevation,
# station depth


def process_unit(tr_add, target_path, input_dics, staev_ar):
    """
    processing unit, adjustable by the user
    :param tr_add: address of one trace in your dataset. You can use that to
    read in the data.
    :param target_path: address of the event that should be processed.
    :param input_dics: dictionary that contains all the inputs.
    :param staev_ar: an array that contains the following information:
           net, sta, loc, cha, station latitude, station longitude,
           station elevation, station depth
    :return:
    """
    # -------------- read the waveform, deal with gaps ------------------------
    # 1. read the waveform and create an obspy Stream object
    try:
        st = read(tr_add)
    except Exception as error:
        print('WARNING: %s' % error)
        return False
    # 2. in case that there are more than one waveform in a Stream (this can
    # happen due to some gaps in the waveforms) merge them.
    if len(st) > 1:
        try:
            st.merge(method=1, fill_value=0, interpolation_samples=0)
        except Exception as error:
            print('WARNING: %s' % error)
            return False
    # 3. Now, there is only one waveform, create a Trace
    tr = st[0]

    # -------------- path to save the processed waveform ----------------------
    # Before entering to the actual processing part of the code,
    # we define some paths to be used later:
    # you can adjust it as you want, here is just one example
    
    # If pathlib is installed, one can use: (suggested by ghraecakter)
    # pathlib.Path(os.path.join(target_path, 'processed')).mkdir(exist_ok=True)


    if input_dics['sampling_rate'] and input_dics['instrument_correction']:
        file_name_part = 'processed'
    elif input_dics['sampling_rate'] and (not input_dics['instrument_correction']):
        file_name_part = 'proc_resamp'
    elif input_dics['instrument_correction'] and (not input_dics['sampling_rate']):
        file_name_part = 'proc_instr'
    elif (not input_dics['instrument_correction']) and (not input_dics['sampling_rate']):
        file_name_part = 'proc_noinstr_noresamp'
    else:
        return False

    try:
        os.mkdir(os.path.join(target_path, file_name_part))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # save_path is the address that will be used to save the processed data
    save_path = os.path.join(target_path, file_name_part, tr.id)
    if os.path.isfile(save_path) and (not input_dics['force_process']):
        return False

    # -------------- PROCESSING -----------------------------------------------
    # * resample the data
    # input_dics['sampling_rate'] determines the desired sampling rate
    if input_dics['sampling_rate']:
        print("resample %s from %s to %sHz" % (tr.id,
                                               tr.stats.sampling_rate,
                                               input_dics['sampling_rate']))
        tr = resample_unit(tr,
                           des_sr=input_dics['sampling_rate'],
                           resample_method=input_dics['resample_method'])

    # * apply instrument correction which consists of:
    # 1. removing the trend of the trace (remove_trend)
    # 2. remove the mean (zero_mean)
    # 3. taper (taper, taper_fraction [e.g., 0.05: 5%])
    # 4. apply pre-filter based on input_dics['pre_filt']
    # 5. apply instrument correction
    # all the above parameters are adjustable
    if input_dics['instrument_correction']:
        tr = instrument_correction(tr, target_path, save_path,
                                   input_dics['corr_unit'],
                                   input_dics['pre_filt'],
                                   input_dics['water_level'],
                                   zero_mean=True,
                                   taper=True,
                                   taper_fraction=0.05,
                                   remove_trend=True)


    # -------------- OUTPUT ---------------------------------------------------
    if not tr:
        pass
    elif input_dics['waveform_format'] == 'sac':
        tr = convert_to_sac(tr, save_path, staev_ar)
        tr.write(save_path, format='SAC')
    else:
        try:
            tr.write(save_path, format='mseed')
        except Exception as e:
            print('ERROR: %s -- %s' % (save_path, e))

    # -------------- path to save the WAV waveform ----------------------------
    
    # If pathlib is installed, one can use: (suggested by ghraecakter)
    # pathlib.Path(os.path.join(target_path, 'WAV')).mkdir(exist_ok=True)


    if input_dics['sampling_rate'] and input_dics['instrument_correction']:
        wav_file_name_part = 'WAV_processed'
    elif input_dics['sampling_rate'] and (not input_dics['instrument_correction']):
        wav_file_name_part = 'WAV_resamp'
    elif input_dics['instrument_correction'] and (not input_dics['sampling_rate']):
        wav_file_name_part = 'WAV_instr'
    elif (not input_dics['instrument_correction']) and (not input_dics['sampling_rate']):
        wav_file_name_part = 'WAV_noinstr_noresamp'
    else:
        return False
    
    try:
        os.mkdir(os.path.join(target_path, wav_file_name_part))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # import ipdb; ipdb.set_trace()
    # save_path_wav is the address that will be used to save the wav
    save_path_wav = os.path.join(target_path, wav_file_name_part, f'{tr.id}_{tr.stats.starttime.date.year}-{tr.stats.starttime.date.month:02d}-'
                                                                  f'{tr.stats.starttime.date.day:02d}T{tr.stats.starttime.time.hour:02d}-'
                                                                  f'{tr.stats.starttime.time.minute:02d}-{tr.stats.starttime.time.second:02d}.WAV')
    if os.path.isfile(save_path_wav) and (not input_dics['force_process']):
        return False

    # -------------- export WAV trace files -----------------------------------
    framerate = 48000
    bitrate = 'PCM_24'

    with SoundFile(save_path_wav, 'w', samplerate=framerate, channels=1, 
                   subtype=bitrate, endian=None, format=None, closefd=True) as f:
        
        data = tr.data / abs(tr.data).max()
        f.write(data*0.95)
        
        print("Writing WAV file for %s" % tr.id)
    
    f.close() 