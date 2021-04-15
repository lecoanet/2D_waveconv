
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

z_basis_conv2 = de.Chebyshev('z', int(3*N/4), interval=(0,z_match))
z_basis_rad2  = de.Chebyshev('z', int(3*N/2),interval=(z_match,Lz))
z_basis2 = de.Compound('z',[z_basis_conv2,z_basis_rad2])
domain2 = de.Domain([z_basis2], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z_basis_conv_old = de.Chebyshev('z', int(N), interval=(0,z_match))
z_basis_rad_old  = de.Chebyshev('z', int(N/2),interval=(z_match,Lz))
z_basis_old = de.Compound('z',[z_basis_conv_old,z_basis_rad_old])
domain_old = de.Domain([z_basis_old], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

def build_solver(domain, domain_old, N, N_old, k):
    # We have N >= N_old
    # but N/2 < N_old

    T_old = domain_old.new_field()
    T_old.set_scales(1024*3/2/N_old)

    data = np.loadtxt(dir + '/T.dat')
    T_old['g'] = data[1,:]

    T0 = domain.new_field()
    T0['c'][:int(N/2)] = T_old['c'][:int(N/2)]
    T0['c'][int(N/2):int(N/2) + int(N_old/2)] = T_old['c'][N_old:]
#    T0['c'][:N_old] = T_old['c'][:N_old]
#    T0['c'][N:N+int(N_old/2)] = T_old['c'][N_old:]

    rho = domain.new_field()
    T0.set_scales(1)
    rho.require_grid_space()

    np.place(rho['g'], T0['g']>0, -1)
    np.place(rho['g'], T0['g']<=0, S)
    rho['g'] *= T0['g']

    N2 = rho.differentiate(0)

    z = domain.grid(0)

    T0z = T0.differentiate(0)
    T0z['g'][z<0.45] *= 0

    T0z.require_coeff_space()

    logger.info(k)
    problem = de.EVP(domain, variables=['p','T','u','w','Tz','uz'], eigenvalue='om')

    problem.parameters['nu'] = nu
    problem.parameters['kappa'] = kappa
    problem.parameters['T0z'] = T0z
    problem.parameters['S'] = S
    problem.substitutions['dt(A)'] = "-1j*om*A"
    problem.parameters['kx'] = 2*np.pi*k
    problem.substitutions['dx(A)'] = "1j*kx*A"

    problem.add_equation("dx(u) + dz(w) = 0")
    problem.add_equation("dt(T) - kappa*(dx(dx(T)) + dz(Tz)) + w*T0z = 0")
    problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(p) = 0")
    problem.add_equation("dt(w) - nu*(dx(dx(w)) - dz(dx(u))) + dz(p) + S*T = 0")
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("uz - dz(u) = 0")

    problem.add_bc("left(uz)  = 0")
    problem.add_bc("left(w)   = 0")
    problem.add_bc("left(T)   = 0")
    problem.add_bc("right(uz) = 0")
    problem.add_bc("right(w)  = 0")
    problem.add_bc("right(T)  = 0")

    solver = problem.build_solver()
    return solver

def solve_dense(solver):
    solver.solve_dense(solver.pencils[0])

    values = solver.eigenvalues
    vectors = solver.eigenvectors

    cond1 = np.isfinite(values)
    values = values[cond1]
    vectors = vectors[:,cond1]

    cond2 = values.real > 0
    values = values[cond2]
    vectors = vectors[:, cond2]

    order = np.argsort(-values.imag)
    values = values[order]
    vectors = vectors[:, order]

    solver.eigenvalues = values
    solver.eigenvectors = vectors
    return solver

def solve_sparse(solver, eigenvalue):
    solver.solve_sparse(solver.pencils[0], 1, eigenvalue)

    return solver

def check_eigen(solver1, i_eig, solver2):
    value1 = solver1.eigenvalues[i_eig]
    value2 = solver2.eigenvalues[0]

    w1 = solver1.state['w']
    w2 = solver2.state['w']
    w1.set_scales(3/2)

    cutoff1 = 3e-4
    cutoff2 = np.sqrt(cutoff1)
    keep = []
    dist = []

    d = np.min(np.abs(value1-value2)/np.abs(value1))
    if d < cutoff1:
        solver1.set_state(i_eig)
        solver2.set_state(0)

        ix1 = np.argmax(np.abs(w1['g']))
        w1['g'] /= w1['g'][ix1]
        ix2 = np.argmax(np.abs(w2['g']))
        w2['g'] /= w2['g'][ix2]

        vector_diff = np.max(np.abs(w1['g'] - w2['g']))
        if vector_diff < cutoff2:
            return True
    return False

int_field = domain1.new_field()
int_op = de.operators.integrate(int_field, 'z')
def IP(w1, w2, u1, u2):
    int_field.set_scales(1)
    int_field['g'] = (w1*np.conj(w2) + u1*np.conj(u2))
    return int_op.evaluate()['g'][0]

def calculate_duals(w_list, u_list):

    n_modes = w_list.shape[0]
    IP_matrix = np.zeros((n_modes, n_modes),dtype=np.complex128)
    for i in range(n_modes):
        if i % 10 == 0: logger.info("{}/{}".format(i,n_modes))
        for j in range(n_modes):
            IP_matrix[i,j] = IP(w_list[i],w_list[j],u_list[i],u_list[j])

    IP_inv = np.linalg.inv(IP_matrix)

    w_dual = np.einsum('ij,ik->kj',np.array(w_list), np.conj(IP_inv))
    u_dual = np.einsum('ij,ik->kj',np.array(u_list), np.conj(IP_inv))

    return w_dual, u_dual

k_list = np.linspace(1, 20, 20)

for k in k_list[rank::size]:

    solver1 = build_solver(domain1, domain_old, N, N, k)
    solver1 = solve_dense(solver1)
    logger.info('done with first solve for k={}'.format(k))

    values = []
    u1_list = []
    w1_list = []
    u2_list = []
    w2_list = []
    solver2 = build_solver(domain2, domain_old, int(3*N/2), N, k)
    for i, value in enumerate(solver1.eigenvalues):
        if i % 10 == 0: logger.info("{}/{}".format(i, len(solver1.eigenvalues)))
        solve_sparse(solver2, value)
        if check_eigen(solver1, i, solver2):
            solver1.set_state(i)
            solver2.set_state(0)
            u1 = solver1.state['u']
            w1 = solver1.state['w']
            w1.set_scales(1)
            u2 = solver2.state['u']
            w2 = solver2.state['w']
            values.append(value)
            ix = np.argmax(np.abs(w1['g']))
            w0 = w1['g'][ix]
            w1['g'] /= w0
            u1['g'] /= w0
            ix = np.argmax(np.abs(w2['g']))
            w0 = w2['g'][ix]
            w2['g'] /= w0
            u2['g'] /= w0
            u1_list.append(np.copy(u1['g']))
            w1_list.append(np.copy(w1['g']))
            u2_list.append(np.copy(u2['g']))
            w2_list.append(np.copy(w2['g']))
    u1_list = np.array(u1_list)
    w1_list = np.array(w1_list)
    u2_list = np.array(u2_list)
    w2_list = np.array(w2_list)
    values = np.array(values)

    logger.info('done with second solve for k={}'.format(k))

    z1 = domain1.grid(0, scales=1)
    z2 = domain2.grid(0, scales=1)
    data = {'u1': u1_list, 'u2': u2_list, 'w1': w1_list, 'w2': w2_list, 'values': values, 'z1': z1, 'z2': z2}
    pickle.dump(data, open('eigenvalues/eigenvalue_data_raw_k%i.pkl' %k,'wb'))

    w_dual, u_dual = calculate_duals(w1_list, u1_list)

    data = {'u_dual': u_dual, 'values': values, 'u': u1_list, 'w': w1_list, 'u2': u2_list, 'w2': w2_list, 'z':z1}

    pickle.dump(data, open('eigenvalues/eigenvalue_data_k%i.pkl' %k,'wb'))


