"""
Calculate the average temperature profile

Usage:
    ave_T.py <files>... [options]

Options:
    --output=<output>    Output filename, default based off file name.
    --plot=<plot>        If True, plots the temperature profile. [default: False]
"""

import numpy as np
import h5py

from dedalus.tools import logging as logging_setup
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

files = args['<files>']

T_list = []

for i, file in enumerate(files):
    if i % 10 == 0:
        logger.info(i)
    f = h5py.File(file)

    z = np.array(f['scales/z/1.0'])
    T = np.array(f['tasks/T ave'][:,0,:])
    T_list.append(T)

T = np.vstack(T_list)
T_ave = np.mean(T, axis=0)

data = np.vstack((z,T_ave))

dir = files[0].split('/')[0]
if args['--output']:
    filename = '%s/%s' %(dir, args['--output'])
else:
    filename = '%s/T.dat' %( dir )

logger.info(filename)
np.savetxt(filename, data)

if args['--plot']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(T_ave, z)

    plt.xlabel('T')
    plt.ylabel('z')

    plt.savefig('%s/T.png' %dir, dpi=150)
