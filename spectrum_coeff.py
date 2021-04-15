"""
Calculate spatial fourier coefficients from data on a vertical level.

Usage:
    spectrum_coeff.py <files>... [options]

Options:
    --output=<output>    Way to append file to denote spectrum, default is _c
"""

import numpy as np
import h5py
import dedalus.public as de

import logging
logger = logging.getLogger(__name__)

from collections import OrderedDict
from dedalus.tools.general import natural_sort
import pathlib
import os

from mpi4py import MPI

from docopt import docopt
args = docopt(__doc__)

comm = MPI.COMM_SELF

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

files = natural_sort(args['<files>'])
f = h5py.File(files[0])

x = np.array(f['scales/x/1.0'])
N = len(x)
logger.info(N)

Lx, Lz = (1, 1)

x_basis = de.Fourier('x', N, interval=(-Lx/2, Lx/2))
domain = de.Domain([x_basis], grid_dtype=np.float64, comm=comm)

data = domain.new_field()

kx = domain.elements(0)[:]
logger.info(kx.shape)

for file in files[rank::size]:
    logger.info(file)

    # transform data
    f = h5py.File(file)
    t = np.array(f['scales/sim_time'])

    spectra = OrderedDict()

    for task in f['tasks']:
        logger.info(task)
    
        if task[2] == 'z':

            spectrum = np.zeros((len(t),len(kx)),dtype=np.complex128)

            for i in range(len(t)):
                data['g'] = np.array(f['tasks/%s' %task][i,:,0])

                spectrum[i] = data['c']*np.sqrt(2*N)

            spectra[task] = spectrum

            energy_g = np.sum(data['g']**2)
            energy_c = np.sum(np.abs(spectrum[i][1:])**2) + 0.5*np.abs(spectrum[i][0])**2
            logger.info(energy_g)
            logger.info(energy_c)

    f.close()

    # write data
    index_under = file.rfind('_')
    if args['--output'] is None:
        addition = '_c'
    else:
        addition = args['--output']
    new_file = file[:index_under] + addition + file[index_under:]
    logger.info(new_file)

    spectra_file = pathlib.Path(new_file).absolute()
    if os.path.exists(str(spectra_file)):
        spectra_file.unlink()
    spectra_f = h5py.File('{:s}'.format(str(spectra_file)), 'a')
    scale_group = spectra_f.create_group('scales')
    scale_group.create_dataset(name='sim_time', data = t)
    scale_group.create_dataset(name='kx', data = kx)

    task_group = spectra_f.create_group('tasks')
    for key in spectra.keys():
        task_group.create_dataset(name=key, data = spectra[key])

    spectra_f.close()

