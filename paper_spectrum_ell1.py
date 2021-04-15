
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

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot ) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes = fig.add_axes([left, bottom, width, height])

# load data
file = 'Ra1e10/slices_rst2/slices_rst2_c_freq.h5'
N = np.sqrt(100*200/0.5)/(2*np.pi)

f = h5py.File(file)
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

k = 1

ux2 = ux2[:,k]
ux = np.sqrt(ux2)

data = pickle.load(open("Ra1e10/eigenvalues/transfer_k1.pkl",'rb'))

om = data['om']
freq_T = om/(2*np.pi)
transfer = data['transfer']/(2*np.pi) # forgot a 2 pi in the definition of the transfer function

ur = np.sqrt(2*k/np.sqrt(N**2-freq_T**2))*np.sqrt(2e-5)*(freq_T)**(-13/4)*(k)**(3/2)

ratio = np.loadtxt("Ra1e10/eigenvalues/ratio_k1.txt")
data_eig = pickle.load(open("Ra1e10/eigenvalues/eigenvalue_data_k1.pkl",'rb'))
values = data_eig['values']
gamma = -values.imag
freq_eig = values.real/(2*np.pi)

freq_next = np.copy(freq_eig)
freq_next[0] = N
freq_next[1:] = freq_eig[:-1]
freq_spacing = freq_next - freq_eig

mask = gamma/freq_spacing < 1
ratio = ratio[mask]
gamma = gamma[mask]
freq_eig = freq_eig[mask]
freq_spacing = freq_spacing[mask]

spec = 2e-5 * (freq_eig/freq[0]) * (freq_eig)**(-13/2) * (k)**(3)

E_mode = spec*(freq_spacing/freq_eig)*ratio/gamma
ux_mode = 0.3*(freq_eig*1.3)/k * np.sqrt(E_mode)

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

plot_axes.loglog(freq/N, ux, color='MidnightBlue', label='sim')
plot_axes.loglog(freq_T/N, ur*transfer, color='DarkGoldenrod',linewidth=0.5, label='transfer')
plot_axes.scatter(freq_eig/N, ux_mode, color='Firebrick', label='mode', marker='x', linewidth=0.5, s=20)

lg = plot_axes.legend(bbox_to_anchor=(0.1, 0.0), loc='lower left')
lg.draw_frame(False)

plot_axes.set_xlim([1.5e-2,1.5])
plot_axes.set_ylim([1e-7, 1e-1])

plot_axes.text(0.5,1.07,r'$\ell=1$',va='center',ha='center',fontsize=fontsize,transform=plot_axes.transAxes)

plot_axes.set_xlabel(r'$f/N$', fontsize=fontsize)
plot_axes.set_ylabel(r'$|\hat{u}_x|_{z=L}$', fontsize=fontsize)

plot_axes.text(-0.24,1.12,r'$C^{10}$',va='center',ha='left',fontsize=18,transform=plot_axes.transAxes)

plt.savefig('figures/spectrum_ell1.eps')
#plt.savefig('figures/spectrum_ell1.png')

