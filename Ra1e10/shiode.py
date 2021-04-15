
import numpy as np
import h5py
from scipy import interpolate
import pickle

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

dir = 'profiles'

# Parameters
Lx, Ly, Lz = (1, 1, 1.)
z_match = 0.6

Rayleigh = 1e10*8
Prandtl = 1.
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)

S = 100
T_top = -200

N2 = np.abs(S*T_top*2)
f = np.sqrt(N2)/(2*np.pi)

N = int(256)
# Create bases and domain
z_basis_conv1 = de.Chebyshev('z', int(N/2), interval=(0,z_match))
z_basis_rad1  = de.Chebyshev('z', int(N),interval=(z_match,Lz))
z_basis1 = de.Compound('z',[z_basis_conv1,z_basis_rad1])
domain1 = de.Domain([z_basis1], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

int_field = domain1.new_field()
u_field = domain1.new_field()

ratio = []
for k in range(1, 21):
    data = pickle.load(open('eigenvalues/eigenvalue_data_k%i.pkl' %k,'rb'))

    u = data['u']
    w = data['w']
    freq = data['values']

    ratio = []
    for i in range(u.shape[0]):

        int_field['g'] = (u[i]*np.conj(u[i]) + w[i]*np.conj(w[i]))
        KE = de.operators.integrate(int_field, 'z').evaluate()['g'][0]
        u_field['g'] = u[i]
        ux2 = np.abs(de.operators.interpolate(u_field, z=1).evaluate()['g'][0])**2

        ratio.append(np.real(ux2/KE))

    np.savetxt('eigenvalues/ratio_k%i.txt' %k, ratio)

