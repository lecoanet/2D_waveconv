"""
Calculate temporal fourier transform.

Usage:
    spectrum_freq.py <files>... [options]

Options:
    --start=<start>      Start index of file list, default 0.
    --end=<end>          End index of file list, default -1.
    --output=<output>    Output filename, default based off file name.
"""

import numpy as np
import h5py
from dedalus.tools.general import natural_sort

import pathlib
import os

from scipy import signal

from docopt import docopt

from dedalus.tools import logging as logging_setup
import logging
logger = logging.getLogger(__name__)

def frequency_spectrum(task, files):

    data_list = []
    time_list = []

    for file in files:
        f = h5py.File(file)

        time_list.append(np.array(f['scales/sim_time']))
        data_list.append(np.array(f['tasks/%s' %task]))
        f.close()

    time = np.hstack(time_list)
    logger.info(time[0])
    logger.info(time[-1])
    data = np.vstack(data_list)

    t_len = len(time)
    window = signal.hann(t_len)

    shape = list(data.shape)
    for i in range(1,data.ndim):
        shape[i] = 1

    data_norm = np.sum(np.abs(data)**2,axis=0)

    window = window.reshape(shape)
    data *= window

    data_new_norm = np.sum(np.abs(data)**2,axis=0)
    # renormalize data because of window function
    data *= np.sqrt( data_norm/data_new_norm )
    logger.info( np.sqrt( data_norm/data_new_norm )[1:4] )

    spectrum_freq = np.fft.fft(data,axis=0)/t_len
    freq = np.fft.fftfreq(t_len,d = time[-1]-time[-2])

    energy_g = np.sum(np.abs(data)**2,axis=0)/t_len # average energy
    energy_c = np.sum(np.abs(spectrum_freq)**2,axis=0) # should also be average energy
    logger.info(energy_g[1])
    logger.info(energy_c[1])

    return freq, time, spectrum_freq

def calculate_spectrum(files, start, end, output):

    file = files[0]
    f = h5py.File(file)
    tasks = tuple(f['tasks'].keys())
    kx = np.array(f['scales/kx'])
    f.close()

    new_file = output

    logger.info(new_file)

    spectra_file = pathlib.Path(new_file).absolute()
    if os.path.exists(str(spectra_file)):
        spectra_file.unlink()
    spectra_f = h5py.File('{:s}'.format(str(spectra_file)), 'a')
    scale_group = spectra_f.create_group('scales')
    scale_group.create_dataset(name='kx', data = kx)

    task_group = spectra_f.create_group('tasks')
    for task in tasks:
        logger.info(task)
        freq, time, spectrum_freq = frequency_spectrum(task, files[start:end])
        task_group.create_dataset(name=task, data = spectrum_freq)

    scale_group.create_dataset(name='f', data = freq)
    scale_group.create_dataset(name='t', data = time)

    spectra_f.close()

if __name__ == "__main__":

    args = docopt(__doc__)

    files = natural_sort(args['<files>'])

    if args['--start'] == None: start = 0
    else: start = int(args['--start'])

    if args['--end'] == None: end=-1
    else: end = int(args['--end'])

    if args['--output'] is None:
        file = files[0]
        index_under = file.rfind('_')
        new_file = file[:index_under] + '_freq.h5'
    else:
        new_file = args['--output']

    calculate_spectrum(files, start, end, new_file)

