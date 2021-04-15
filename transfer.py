"""
Calculate transfer function to get horizontal velocities at the top of the simulation.

Usage:
    transfer.py <directory>

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate

from dedalus.tools import logging as logging_setup
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

from dedalus import public as de

# Parameters
Lx, Ly, Lz = (1, 1, 1.)
z_match = 0.6

Rayleigh = 1e9*8
Prandtl = 1.
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)

S = 100
T_top = -100

N2 = np.abs(S*T_top*2)
f = np.sqrt(N2)/(2*np.pi)

N = 512
# Create bases and domain
z_basis_conv1 = de.Chebyshev('z', int(N), interval=(0,z_match))
z_basis_rad1  = de.Chebyshev('z', int(N/2),interval=(z_match,Lz))
z_basis1 = de.Compound('z',[z_basis_conv1,z_basis_rad1])
domain1 = de.Domain([z_basis1], grid_dtype=np.complex128)

k_list = np.arange(1, 21)
logger.info(k_list)

def transfer_function(om, values, u_dual, u_outer, k):
    om = om[None, None, :]
    T = (1j*np.sqrt(2)*om/k*u_dual[:,:, None]*u_outer[:, None, None])/(om-values[:, None, None])
    return np.mean(np.abs(np.sum(T, axis=0)), axis=0)

def refine_peaks(om, T, k, u_dual, u_outer):
    i_peaks = []
    for i in range(1,len(om)-1):
        if (T[i]>T[i-1]) and (T[i]>T[i+1]):
            delta_m = np.abs(T[i]-T[i-1])/T[i]
            delta_p = np.abs(T[i]-T[i+1])/T[i]
            if delta_m > 0.05 or delta_p > 0.05:
                i_peaks.append(i)

    logger.info("number of peaks: %i" %(len(i_peaks)))

    om_new = np.array([])
    for i in i_peaks:
        om_low = om[i-1]
        om_high = om[i+1]
        om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])

    T_new = transfer_function(om_new, values, u_dual, u_outer, k)

    om = np.concatenate([om,om_new])
    T = np.concatenate([T,T_new])

    om, sort = np.unique(om, return_index=True)
    T = T[sort]

    return om, T, len(i_peaks)

dir = args['<directory>']

for k in k_list:
    logger.info("k = %i" %k)

    data = pickle.load(open(dir + '/eigenvalues/eigenvalue_data_k%i.pkl' %k,'rb'))
 
    values = data['values']
    u_dual = data['u_dual']
    u = data['u']
    z = data['z']

    om0 = values.real[-2]
    print(om0)
    om1 = values.real[0]*2
    om = np.exp( np.linspace(np.log(om0), np.log(om1), num=1000, endpoint=True) )

    z0 = 0.5
    z1 = 0.6
    z_range = np.linspace(z0, z1, num=100, endpoint=True)
    u_dual_interp = interpolate.interp1d(z, u_dual, axis=-1)(z_range)
    u_z = u[:,-1]

    T = transfer_function(om, values, u_dual_interp, u_z, k)

    peaks = 1
    while peaks > 0:
        om, T, peaks = refine_peaks(om, T, k, u_dual_interp, u_z)

#    if k in [1, 2, 3, 5, 10, 20]:
    if k in range(1, 21):
        data = pickle.load(open(dir + '/forced/forced_k%i.pkl' %k, 'rb'))
        om_f = data['om']
        # need a mysterious factor of 2.....
        T_f = 2*data['transfer']
        cutoff = 2*np.pi*k**(3/4)
        T = np.concatenate((  T_f[om_f < cutoff],  T[om > cutoff]))
        om = np.concatenate((om_f[om_f < cutoff], om[om > cutoff]))

    data = {'om':om, 'transfer':T}
    pickle.dump(data, open(dir + '/eigenvalues/transfer_k%i.pkl' %k,'wb'))

