
import numpy as np
import pickle
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from dedalus.tools import logging as logging_setup
import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI

plot = False

k_list = [19]

for k in k_list:    
    f_min = 0.6*k**(3/4)
    f_max = 2*k**(3/4)
    
    f_list = np.exp(np.linspace(np.log(f_min), np.log(f_max), num=100))
    z_force_list = np.linspace(0.5,0.55,num=11)
    
    u_list = []
    
    for f in f_list:
        logger.info(f)
        u_z = []
        for z_force in z_force_list:
    
            f_str = "%.3f" %f
            f_str = f_str.replace('.','p')
            z_force_str = "%.3f" %z_force
            z_force_str = z_force_str.replace('.','p')
        
            base = "u_top_z%s_f%s" %(z_force_str, f_str)
        
            file = h5py.File('k' + str(k) + '/' + base + '/' + base + '_s1/' + base + '_s1_p0.h5')
        
            u_top = np.array(file['tasks/u'][:,0])
            time = np.array(file['scales/sim_time'])
        
            u_ave = np.mean(np.abs(u_top)[-150:])
        
            if plot:
                plt.plot(time, np.abs(u_top))
                plt.plot(time, u_ave + 0*time)
                plt.savefig(base + '/u_top_f' + f_str + '.png', dpi=150)
                plt.close()
        
            u_z.append(u_ave)
        u_list.append(u_z)
    
    u_list = np.array(u_list)
    logger.info(u_list.shape)
    
    om = 2*np.pi*f_list
    
    T = np.sqrt(2)*om/k*np.mean(u_list, axis=1)/0.01
    
    data = {'transfer': T, 'om': om}
    
    pickle.dump(data, open('forced_k%i.pkl' %k,'wb'))
    
    #plt.loglog(f_list, T)
    #plt.savefig('transfer_k1.png', dpi=150)
    
