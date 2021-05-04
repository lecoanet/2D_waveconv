"""
Calculate temporal fourier transform.
mpiexec_mpt -np 1 python3 spectrogram.py Ra1e10/slices/slices_c_s* Ra1e10/slices_rst/slices_rst_c_s* Ra1e10/slices_rst2/slices_rst2_c_s* --start=3


Usage:
    spectrum_freq.py <files>... [options]

Options:
    --start=<start>      Start index of file list, default 0.
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

def rearrange(time, data, start):

    num_transitions = 2
    dt = np.abs(time[1:] - time[:-1])
    transitions = np.argpartition(dt, -num_transitions)[-num_transitions:]

    data_list = []
    time_list = []
    for i in range(num_transitions+1):
        if i == 0:
            s = slice(0, transitions[i]+1)
        elif i == num_transitions:
            s = slice(transitions[i-1]+2, -1)
        else:
            s = slice(transitions[i-1]+2, transitions[i])
        time_list.append(time[s])
        data_list.append(data[s])

    sort = np.argsort([t[0] for t in time_list])
    time_list = [time_list[i] for i in sort]
    data_list = [data_list[i] for i in sort]

    for j in range(num_transitions):
        i = np.argmin(np.abs(time_list[j+1] - time_list[j][-1]))
        time_list[j+1] = time_list[j+1][i+1:]
        data_list[j+1] = data_list[j+1][i+1:]

    time = np.concatenate(time_list)
    data = np.concatenate(data_list)

    i_start = start*200
    time = time[i_start:]
    data = data[i_start:,:]

    dt = time[1:] - time[:-1]
    logger.info(np.max(dt))
    logger.info(np.min(dt))

    return time, data

def frequency_spectrum(task, files, start):

    data_list = []
    time_list = []

    for file in files:
        f = h5py.File(file)

        time_list.append(np.array(f['scales/sim_time']))
        data_list.append(np.array(f['tasks/%s' %task]))
        f.close()

    time = np.hstack(time_list)
    data = np.vstack(data_list)

    time, data = rearrange(time, data, start)
    logger.info(time[0])
    logger.info(time[-1])

    win_len = 2000
    spectrum_list = []
    time_list = []
    dt = 200
    for i in range(0,len(time)-win_len, dt):
        if i % 100 == 0:
            logger.info(i)
        window = signal.hann(win_len)
        window = window.reshape((win_len, 1))

        data_norm = np.sum(np.abs(data[i:i+win_len])**2,axis=0)

        data_win = data[i:i+win_len]*window

        data_new_norm = np.sum(np.abs(data_win)**2,axis=0)
        # renormalize data because of window function
        data_win *= np.sqrt( data_norm/data_new_norm )
        #logger.info( np.sqrt( data_norm/data_new_norm )[1:4] )

        spectrum_freq = np.fft.fft(data_win,axis=0)/win_len
        freq = np.fft.fftfreq(win_len,d = time[-1]-time[-2])

        spectrum_list.append(spectrum_freq)
        time_list.append(time[i+win_len//2])

        #energy_g = np.sum(np.abs(data_win)**2,axis=0)/win_len # average energy
        #energy_c = np.sum(np.abs(spectrum_freq)**2,axis=0) # should also be average energy
        #logger.info(energy_g[1])
        #logger.info(energy_c[1])

    time = np.array(time_list)
    spectrum = np.array(spectrum_list)
    logger.info('done')
    return freq, time, spectrum

def calculate_spectrum(files, start, output):

    file = files[0]
    f = h5py.File(file)
    tasks = tuple(f['tasks'].keys())
    kx = np.array(f['scales/kx'])
    f.close()

    new_file = output

    spectra_file = pathlib.Path(new_file).absolute()
    if os.path.exists(str(spectra_file)):
        spectra_file.unlink()
    spectra_f = h5py.File('{:s}'.format(str(spectra_file)), 'a')
    scale_group = spectra_f.create_group('scales')
    scale_group.create_dataset(name='kx', data = kx)

    task_group = spectra_f.create_group('tasks')
    for task in ['u z=1.0']:
        logger.info(task)
        freq, time, spectrum_freq = frequency_spectrum(task, files, start)
        task_group.create_dataset(name=task, data = spectrum_freq)

    scale_group.create_dataset(name='f', data = freq)
    scale_group.create_dataset(name='t', data = time)

    spectra_f.close()

if __name__ == "__main__":

    args = docopt(__doc__)

    files = natural_sort(args['<files>'])

    logger.info(files)

    if args['--start'] == None: start = 0
    else: start = int(args['--start'])

    if args['--output'] is None:
        file = files[0]
        index_under = file.rfind('_')
        new_file = file[:index_under-2] + '_spectrogram.h5'
    else:
        new_file = args['--output']

    calculate_spectrum(files, start, new_file)

