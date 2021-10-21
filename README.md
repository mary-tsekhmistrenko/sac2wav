# sac2wav
Downloads seismic waveforms and converts them to WAV files. 

## Set up
The easiest way to run the code is by:

    1. Setting up Anaconda and creating an environment (e.g. sounds) with Python 3
    2. Download the necessary packages
    3. Download obspyDMT (see below for more information)
    4. run the jupyter notebook from your newly created environment


## How to install Anaconda?

I found this website very helpful when installing Anaconda and setting up environments:
https://docs.anaconda.com/anaconda/install/mac-os/



## What packages will you need?

The usual suspects:

    Anaconda
    obspy
    numpy, scipy, ...
    cartopy 

The extras:

    SoundFile
    natsorted

Don't worry, if you miss something the code will tell you what is missing.


## How to set up obspyDMT?

Download obspyDMT into a local floder:

    git clone https://github.com/kasra-hosseini/obspyDMT.git /path/to/my/obspyDMT

obspyDMT can be installed by:

    cd /path/to/my/obspyDMT
    pip install -e .

Please do not use >>pip install obspyDMT<<.

Copy process_unit_wav.py which you can find in this project folder into:

    /path/to/my/obspyDMT/obspyDMT

You will find there also some other files named process_unitXXX.py in this path.

Learn more about obspyDMT and how it works:

https://github.com/kasra-hosseini/obspyDMT


## Useful links

Websites for finding earthquakes, networks and stations:

- http://www.fdsn.org/networks/
- http://ds.iris.edu/wilber3/find_event

Understanding network codes:

- https://ds.iris.edu/ds/nodes/dmc/tools/data_channels/#???
- https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/

Learn more about python, jupyter notebooks and obspy:

- http://seismo-live.org/

What we are using to export the waveforms to WAV files:

- https://pysoundfile.readthedocs.io/en/latest/