
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

t_mar, b_mar, l_mar, r_mar = (0.24, 0.4, 0.6, 0.5)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)
h_pad = 0.1
w_pad = 0.1

h_total = t_mar + 4*h_plot + 3*h_pad + b_mar
w_total = l_mar + 3*w_plot + 2*w_pad + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
plot_axes = []
for i in range(3):
    for j in range(4):
        left = (l_mar + i*w_plot + i*w_pad) / w_total
        bottom = 1 - (t_mar + (j+1)*h_plot + j*h_pad ) / h_total
        width = w_plot / w_total
        height = h_plot / h_total
        plot_axes.append( fig.add_axes([left, bottom, width, height]) )

# load data
file_list = ['Ra2e8/slices/slices_c_freq.h5', 'Ra1e9/slices_rst2/slices_rst2_c_freq.h5', 'Ra1e10/slices_rst2/slices_rst2_c_freq.h5']
#file_list = ['Ra2e8/slices/slices_c_freq.h5', 'Ra1e9/stampede/slices_c_freq.h5', 'Ra1e10/slices_rst2/slices_rst2_c_freq.h5']
#file_list = ['Ra2e8/slices/slices_c_freq.h5', 'Ra1e9/slices_rst/slices_rst_c_freq.h5', 'Ra1e10/slices_rst2/slices_rst2_c_freq.h5']
N_list = [np.sqrt(60*100/0.5)/(2*np.pi), np.sqrt(100*100/0.5)/(2*np.pi),  np.sqrt(100*200/0.5)/(2*np.pi)]

kx_list = [1, 2, 5, 10]

freq_list = []
ux_list = []

for i in range(3):
    for j in range(4):
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

        ux2 = ux2[:,kx_list[j]]
        ux_list.append(np.sqrt(ux2))
    freq_list.append(freq)

dir_list = ['Ra2e8', 'Ra1e9', 'Ra1e10']

fudge_list = [0.4, 4, 1]
amp_list = [1e-5, 2e-6, 2e-5]
a_list = [3, 3, 3]
b_list = [-15/2,-13/2,-13/2]

#mode_fudge = [0.005, 5, 0.2]
mode_fudge = [0.05, 1.5, 0.3]
tau_list = [1.19, 1.27, 1.3]
freq_T_list = []
ux_T_list = []
mode_amp_list = []
freq_eig_list = []

for i in range(3):
    for j in range(4):
        data = pickle.load(open(dir_list[i] + "/eigenvalues/transfer_k%i.pkl" %kx_list[j],'rb'))
        ratio = np.loadtxt(dir_list[i] + "/eigenvalues/ratio_k%i.txt" %kx_list[j])
        data_eig = pickle.load(open(dir_list[i] + "/eigenvalues/eigenvalue_data_k%i.pkl" %kx_list[j],'rb'))
        values = data_eig['values']
        gamma = -values.imag
        freq_eig = values.real/(2*np.pi)

        om = data['om']
        freq_T = om/(2*np.pi)
        transfer = data['transfer']/(2*np.pi) # forgot 2pi in the calculation of T

        ur = fudge_list[i]*np.sqrt(2*amp_list[i]*kx_list[j]/np.sqrt(N_list[i]**2-freq_T**2))*(freq_T)**(b_list[i]/2)*(kx_list[j])**(a_list[i]/2)
        freq_T_list.append(freq_T)
        ux_T_list.append(ur*transfer)

        freq_next = np.copy(freq_eig)
        freq_next[0] = N_list[i]
        freq_next[1:] = freq_eig[:-1]
        freq_spacing = freq_next - freq_eig

        mask = gamma/freq_spacing < 1
        ratio = ratio[mask]
        gamma = gamma[mask]
        freq_eig = freq_eig[mask]
        freq_spacing = freq_spacing[mask]

        spec = amp_list[i] * (freq_eig/freq_list[i][0]) * (freq_eig)**(b_list[i]) * (kx_list[j])**(a_list[i])
#        spec = amp_list[i] * (freq_eig)**(b_list[i]) * (kx_list[j])**(a_list[i])

        E_mode = spec*(freq_spacing/freq_eig)*ratio/gamma
#        E_mode = 10*spec*(freq_spacing/freq_eig)*ratio/gamma
#        E_mode = mode_fudge[i]*spec/kx_list[j]*ratio/gamma
        ux_mode = mode_fudge[i]*(freq_eig*tau_list[i])/kx_list[j] * np.sqrt(E_mode)
        freq_eig_list.append(freq_eig)
        mode_amp_list.append(ux_mode)

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

for i in range(3):
    for j in range(4):
        ax = plot_axes[4*i+j]

        ax.loglog(freq_list[i]/N_list[i], ux_list[4*i+j], color='MidnightBlue', label='sim')
        ax.loglog(freq_T_list[4*i+j]/N_list[i], ux_T_list[4*i+j], color='DarkGoldenrod',linewidth=0.5, label='transfer')
        ax.scatter(freq_eig_list[4*i+j]/N_list[i], mode_amp_list[4*i+j], color='Firebrick', marker='x', linewidth=0.5, s=20, label='mode')

        if i == 0 and j == 3:
            lg = ax.legend(loc='upper left')
            lg.draw_frame(False)

        ax.set_xlim([1.5e-2,1.5])
        ax.set_ylim([1e-7, 1e-1])

for i in range(3):
    for j in range(4):
        ax = plot_axes[4*i+j]
        if i>0:
            plt.setp(ax.get_yticklabels(), visible=False)
        if j < 3:
            plt.setp(ax.get_xticklabels(), visible=False)


class CustomLogFormatter(LogFormatterMathtext):

    def __call__(self, x, pos = None):

        if x < 0.01:
            return LogFormatterMathtext.__call__(self, x, pos = None)

        else:
            return ' '


for i in range(3):
  for j in range(4):
    ax = plot_axes[4*i+j]
    if i == 0:
      if j > 0:
        ax.yaxis.set_major_formatter(CustomLogFormatter())

sim_names = [r'$C^8$', r'$C^9$', r'$C^{10}$']
for i in range(3):
    ax = plot_axes[4*i]
    ax.text(0.5,1.1,sim_names[i],va='center',ha='center',fontsize=fontsize,transform=ax.transAxes)

for j in range(4):
    ax = plot_axes[8+j]
    ax.text(1.04,0.5,r'$\ell=%i$' %kx_list[j],va='center',ha='left',fontsize=fontsize,transform=ax.transAxes)

for i in range(3):
    ax = plot_axes[4*i+3]
    ax.set_xlabel(r'$f/N$', fontsize=fontsize)

for j in range(4):
    ax = plot_axes[j]
    ax.set_ylabel(r'$|\hat{u}_x|_{z=L}$', fontsize=fontsize)

plt.savefig('figures/spectrum_all.eps')
#plt.savefig('figures/spectrum_all.png')

