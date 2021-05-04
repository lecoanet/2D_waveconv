
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import h5py
import publication_settings
import sys
import pickle

matplotlib.rcParams.update(publication_settings.params)

fontsize = 12

t_mar, b_mar, l_mar, r_mar = (0.24, 0.4, 0.45, 0.1)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)
h_pad = 0.1

h_total = t_mar + 3*h_plot + 2*h_pad + b_mar
w_total = l_mar + w_plot + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
plot_axes = []
for i in range(3):
    left = (l_mar) / w_total
    bottom = 1 - (t_mar + (i+1)*h_plot + i*h_pad ) / h_total
    width = w_plot / w_total
    height = h_plot / h_total
    plot_axes.append( fig.add_axes([left, bottom, width, height]) )

# load data
file_list = ['Ra2e8/slices/slices_c_freq.h5', 'Ra1e9/slices_rst2/slices_rst2_c_freq.h5', 'Ra1e10/slices_rst2/slices_rst2_c_freq.h5']
N_list = [np.sqrt(60*100/0.5)/(2*np.pi), np.sqrt(100*100/0.5)/(2*np.pi),  np.sqrt(100*200/0.5)/(2*np.pi)]

k_max = [20, 20, 20]

freq_list = []
ux_list = []

for i in range(3):
    f = h5py.File(file_list[i])
    freq = np.array(f['scales/f'])
    ux = np.array(f['tasks/u z=1.0'])
    f.close()

    ux2 = np.real(ux*np.conj(ux))

    # add positive and negative frequency
    nf = ux2.shape[0]
    mid_f = int((nf-1)/2)
    if nf % 2 == 0:
        ux2 = ux2[1:mid_f+1] + ux2[-1:mid_f+1:-1]
        freq = freq[1:mid_f+1]
    else:
        ux2 = ux2[1:mid_f+1] + ux2[-1:mid_f:-1]
        freq = freq[1:mid_f+1]

    ux2 = np.sum(ux2[:,1:k_max[i]+1], axis=1)
    ux_list.append(np.sqrt(ux2))
    freq_list.append(freq)

dir_list = ['Ra2e8', 'Ra1e9', 'Ra1e10']

fudge_list = [0.4, 4, 1]
amp_list = [1e-5, 2e-6, 2e-5]
a_list = [3, 3, 3]
b_list = [-15/2,-13/2,-13/2]

freq_T_list = []
ux_T_list = []

for i in range(3):
    freq_T = []
    for k in range(1,k_max[i]+1):
        data = pickle.load(open(dir_list[i] + "/eigenvalues/transfer_k%i.pkl" %k,'rb'))
        om = data['om']
        freq_T = freq_T + list(om/(2*np.pi))
    freq_T = np.array(freq_T)
    freq_T = np.unique(freq_T)
    freq_T_list.append(freq_T)

    transfer = 0*freq_T
    ux2 = 0*freq_T
    for k in range(1,k_max[i]+1):
        data = pickle.load(open(dir_list[i] + "/eigenvalues/transfer_k%i.pkl" %k,'rb'))
        om = data['om']
        freq_k = om/(2*np.pi)
        transfer_k = data['transfer'] / (2*np.pi) # missed factor of 2pi in calculation of T
        mask = (freq_T >= freq_k[0]) * (freq_T <= freq_k[-1])
        freq_subset = freq_T[mask]
        transfer_interp = np.interp(freq_subset, freq_k, transfer_k)

        ur = fudge_list[i]*np.sqrt(2*amp_list[i]*k/np.sqrt(N_list[i]**2-freq_subset**2))*(freq_subset)**(b_list[i]/2)*(k)**(a_list[i]/2)
        ux2[mask] += (ur*transfer_interp)**2


    ux_T_list.append(np.sqrt(ux2))

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

for i in range(3):
    ax = plot_axes[i]
    ax.loglog(freq_list[i]/N_list[i], ux_list[i], color='MidnightBlue', label='sim')
    ax.loglog(freq_T_list[i]/N_list[i], ux_T_list[i], color='DarkGoldenrod',linewidth=0.5, label='transfer')

    ax.set_xlim([1.5e-2,1.5])
    ax.set_ylim([1e-6, 1e-1])

for i in range(3):
    if i < 2:
        plt.setp(plot_axes[i].get_xticklabels(), visible=False)

lg = plot_axes[0].legend(loc='upper left', ncol=2)
lg.draw_frame(False)

plot_axes[2].set_xlabel(r'$2\pi\, f/N$', fontsize=fontsize)

for i in range(3):
    plot_axes[i].set_ylabel(r'$|\hat{u}_x|_{z=1}$', fontsize=fontsize)

sim_num = [8, 9, 10]

for i in range(3):
    ax = plot_axes[i]
    ax.text(0.95,0.87,r'$C^{%i}$' %sim_num[i],va='center',ha='right',fontsize=16,transform=ax.transAxes)

plt.savefig('figures/spectrum_freq.eps')
#plt.savefig('figures/spectrum_freq.png')

