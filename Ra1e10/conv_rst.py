
import numpy as np
from mpi4py import MPI
import time
import h5py
from scipy.special import lambertw

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (1, 1)
z_match = 0.6

Rayleigh = 1e10*8
Prandtl = 1.
nu = 1/np.sqrt(Rayleigh/Prandtl)
kappa = 1/np.sqrt(Rayleigh*Prandtl)

S = 100
T_top = -200

N2 = np.abs(S*T_top*2)
f = np.sqrt(N2)/(2*np.pi)

N = int(1024*3/2)

# Create bases and domain
x_basis = de.Fourier('x', N, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis_conv = de.Chebyshev('z', N, interval=(0,z_match), dealias=3/2)
z_basis_rad  = de.Chebyshev('z', int(N/2),interval=(z_match,Lz), dealias=3/2)
z_basis = de.Compound('z',[z_basis_conv,z_basis_rad],dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

z = domain.grid(1, scales=domain.dealias)

alpha = domain.new_field()
alpha.set_scales(domain.dealias)
alpha.require_grid_space()

heaviside = domain.new_field()
heaviside.set_scales(domain.dealias)

heaviside['g'][:,z[0,:]<0.52] = 1.


def rho_function(*args): # args[0] is T
    np.place(alpha.data,args[0].data>0,-1)
    np.place(alpha.data,args[0].data<=0,S)
    return alpha.data*args[0].data

def rho(*args,domain=domain,F=rho_function):
    return de.operators.GeneralFunction(domain,layout='g',func=rho_function,args=args)

de.operators.parseables['rho'] = rho

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','T','u','w','Tz','uz'])

problem.parameters['Lx'] = Lx
problem.parameters['nu'] = nu
problem.parameters['kappa'] = kappa
problem.parameters['heaviside'] = heaviside
problem.parameters['T_top'] = T_top

problem.add_equation("dx(u) + dz(w) = 0")
problem.add_equation("dt(T) - kappa*(dx(dx(T)) + dz(Tz)) = - (u*dx(T) + w*Tz)")
problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(p) = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - nu*(dx(dx(w)) - dz(dx(u))) + dz(p) = -(u*dx(w) - w*dx(u)) - rho(T)")
problem.add_equation("Tz - dz(T) = 0")
problem.add_equation("uz - dz(u) = 0")

problem.add_bc("left(uz)  = 0")
problem.add_bc("left(w)   = 0")
problem.add_bc("left(T)   = 1")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(w)  = 0", condition = "(nx != 0)")
problem.add_bc("right(p)  = 0", condition = "(nx == 0)")
problem.add_bc("right(T)  = T_top")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
T  = solver.state['T']
Tz = solver.state['Tz']

x = domain.grid(0, scales=3/2)
z = domain.grid(1, scales=3/2)

# Initial conditions
f = h5py.File('checkpoints/checkpoints_s11.h5')

cslices = domain.dist.coeff_layout.slices(scales=(1,1))

for field in solver.state.fields:
    field['c'] = np.array(f['tasks'][field.name][0,cslices[0],cslices[1]])

solver.sim_time = f['scales/sim_time'][0]
solver.iteration = f['scales/iteration'][0]
dt = f['scales/timestep'][0]
f.close()

max_dt = 0.0005

# Integration parameters
solver.stop_sim_time = 100001.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

slice_cadence = 0.01
#slice_time = solver.sim_time+slice_cadence
slice_time = np.ceil(solver.sim_time+1)
#slice_time = 1
slice_num = 0
slice_process = False

ending = '_rst'

# Analysis
slices = solver.evaluator.add_file_handler('slices%s' %ending, sim_dt=1000000, max_writes=200)

for loc in [0.4,0.5,0.6,0.7,0.8,0.9,1]:
    slices.add_task('interp(u, z=%f)' %loc,name='u z=%.1f' %loc)
    slices.add_task('interp(w, z=%f)' %loc,name='w z=%.1f' %loc)
    slices.add_task('interp(p, z=%f)' %loc,name='p z=%.1f' %loc)

profiles = solver.evaluator.add_file_handler('profiles%s' %ending, sim_dt=0.1, max_writes=100)
profiles.add_task("integ(w*T,'x')/Lx",name='conv flux')
profiles.add_task("integ(kappa*dz(T),'x')/Lx",name='diff flux')
profiles.add_task("integ(T,'x')/Lx",name='T ave')
profiles.add_task("integ(u,'x')/Lx",name='u ave')

snapshots = solver.evaluator.add_file_handler('snapshots%s' %ending, sim_dt=0.1, max_writes=40)
snapshots.add_task('T')
snapshots.add_task('rho(T)')
snapshots.add_task('u')
snapshots.add_task('w')

checkpoints = solver.evaluator.add_file_handler('checkpoints%s' %ending, sim_dt=10, max_writes=1)
checkpoints.add_system(solver.state,layout='c')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / nu", name='Re')

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=3, safety=0.35, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u*heaviside', 'w*heaviside'))

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:

        dt = CFL.compute_dt()

        if solver.sim_time > 1:
          dt = min(dt,max_dt)

        t_future = solver.sim_time + dt
        if t_future > slice_time*(1+1e-8):
            dt = slice_time - solver.sim_time
            slice_time += slice_cadence
            slice_num += 1
            slice_process = True
            logger.info(dt)
        elif t_future > slice_time*(1-1e-8):
            slice_time += slice_cadence
            slice_num += 1
            slice_process = True

        solver.step(dt)

        if dt < 1e-10: break
    
        if slice_process:
            slice_process = False
            wall_time = solver.get_world_time() - solver.start_time
            solver.evaluator.evaluate_handlers([slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = solver.get_world_time(),timestep=dt)

        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
