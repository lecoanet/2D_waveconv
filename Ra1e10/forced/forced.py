
import numpy as np
import h5py
from scipy import interpolate
import pickle
import sys
import os

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Parameters
Lx, Ly, Lz = (1, 1, 1.)
z_match = 0.6

Rayleigh = 1e10*8
Prandtl = 1.
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)

S = 100
T_top = -200

k = int(sys.argv[1])

if rank == 0:
    if not os.path.isdir('k%i' %k):
        os.mkdir('k%i' %k)
MPI.COMM_WORLD.Barrier()

# want f_min(k) and f_max(k)
f_min = 0.5*k**(3/4)
f_max = 1*k**(3/4)

f_list = np.exp(np.linspace(np.log(f_min), np.log(f_max), num=100))
logger.info(f_list)

N = int(512)
# Create bases and domain
z_basis_conv = de.Chebyshev('z', int(N), interval=(0,z_match))
z_basis_rad  = de.Chebyshev('z', int(N/2),interval=(z_match,Lz))
z_basis = de.Compound('z',[z_basis_conv,z_basis_rad], dealias=3/2)
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

T0 = domain.new_field()
T0.set_scales(1024*3/2/N)

data = np.loadtxt('T.dat')
T0['g'] = data[1,:]

T0.set_scales(1)

z = domain.grid(0)

T0z = T0.differentiate(0)
T0z.set_scales(1)
T0z['g'] *= 0.5*(1 + np.tanh( (z - 0.4)/0.01 ))

T0z.require_coeff_space()

z_force_list = np.linspace(0.5,0.55,num=11)

# assumes size divides 100 evenly?
f = f_list[rank % 100]
#for f in f_list[rank % 100]:
for z_force in z_force_list[rank // 100::size // 100]:
#for z_force in z_force_list[rank // 100:]:
    problem = de.IVP(domain, variables=['p','T','u','w','Tz','uz'])
    
    problem.parameters['nu'] = nu
    problem.parameters['kappa'] = kappa
    problem.parameters['T0z'] = T0z
    problem.parameters['S'] = S
    problem.parameters['kx'] = 2*np.pi*k
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.parameters['om'] = 2*np.pi*f
    problem.parameters['t0'] = 100/(2*np.pi*f)
    problem.parameters['t_ramp'] = 10/(2*np.pi*f)
    problem.parameters['z_force'] = z_force
    problem.substitutions['F'] = "exp(1j*om*t)*0.5*(1+tanh( (t - t0)/t_ramp ) )*0.5*(tanh( (z-z_force+0.005)/0.001 ) - tanh( (z-z_force-0.005)/0.001))"

    problem.add_equation("dx(u) + dz(w) = 0")
    problem.add_equation("dt(T) - kappa*(dx(dx(T)) + dz(Tz)) = - w*T0z")
    problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(p) = F")
    problem.add_equation("dt(w) - nu*(dx(dx(w)) - dz(dx(u))) + dz(p) + S*T = 0")
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("uz - dz(u) = 0")
    
    problem.add_bc("left(uz)  = 0")
    problem.add_bc("left(w)   = 0")
    problem.add_bc("left(T)   = 0")
    problem.add_bc("right(uz) = 0")
    problem.add_bc("right(w)  = 0")
    problem.add_bc("right(T)  = 0")
    
    solver = problem.build_solver(de.timesteppers.RK222)
    
    solver.stop_sim_time = np.inf
    solver.stop_wall_time = np.inf
    solver.stop_iteration = 100000
    
    dt = 1/20/20*np.pi/3
    
    flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
    flow.add_property("sqrt(u*conj(u))", name='u')
    
    f_str = "%.3f" %f
    f_str = f_str.replace('.','p')
    logger.info(f_str)
    
    z_force_str = "%.3f" %z_force
    z_force_str = z_force_str.replace('.','p')
    logger.info(z_force_str)

    u_top = solver.evaluator.add_file_handler("k%i/u_top_z%s_f%s" %(k, z_force_str, f_str), iter=100)
    u_top.add_task("interp(u, z=1)", name='u')
    
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max u = {}'.format(flow.max('u')))

