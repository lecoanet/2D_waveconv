"""
Calculate the average temperature profile

Usage:
    u_rms.py <files>...

"""

import numpy as np
import h5py

from dedalus.tools.general import natural_sort
from dedalus.tools import logging as logging_setup
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

files = natural_sort(args['<files>'])

start = 3

files = files[3:]

u_list = []
w_list = []

for i, file in enumerate(files):
    if i % 10 == 0:
        logger.info(i)
    f = h5py.File(file)

    u = np.array(f['tasks/u z=0.4'][:,:,0])
    w = np.array(f['tasks/w z=0.4'][:,:,0])
    u_list.append(u)
    w_list.append(w)

u = np.vstack(u_list)
w = np.vstack(w_list)

u_rms2 = np.mean(u**2 + w**2)

u_rms = np.sqrt(u_rms2)

logger.info(u_rms)

