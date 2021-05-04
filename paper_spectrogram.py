
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

t_mar, b_mar, l_mar, r_mar = (0.24, 0.4, 0.45, 0.35)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)

h_cbar, w_cbar = (h_plot, 0.05)
w_pad = 0.05

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + w_pad + w_cbar + r_mar



width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plot
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot ) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes = fig.add_axes([left, bottom, width, height])

# cbar
left = (l_mar + w_plot + w_pad) / w_total
width = w_cbar / w_total
cbar_axes = fig.add_axes([left, bottom, width, height])

# load data
file = 'Ra1e10/slices/slices_spectrogram.h5'
N = np.sqrt(100*200/0.5)/(2*np.pi)

f = h5py.File(file)
freq = np.array(f['scales/f'])
t = np.array(f['scales/t'])
ux = np.array(f['tasks/u z=1.0'])
f.close()

ux2 = np.real(ux*np.conj(ux))

# add positive and negative frequency
nf = ux2.shape[1]
mid_f = int((nf-1)/2)
if nf % 2 == 0:
    ux2 = ux2[:,1:mid_f+1] + ux2[:,-1:mid_f+1:-1]
    freq = freq[1:mid_f+1]
else:
    ux2 = ux2[:,1:mid_f+1] + ux2[:,-1:mid_f:-1]
    freq = freq[1:mid_f+1]


ux2 = np.sum(ux2, axis=2)
ux = np.sqrt(ux2)

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

fm, tm = np.meshgrid(freq/N, t/1.3)

pcm = plot_axes.pcolormesh(fm, tm, ux, norm=matplotlib.colors.LogNorm(), vmin=1e-4, vmax=0.3)

plot_axes.set_xscale('log')
plot_axes.set_xlim([1.5e-2,1.5])
plot_axes.set_ylim([min(t)/1.3,max(t)/1.3])

plot_axes.plot([1.5, 1.5],[9, 80], linewidth=2, color='white')
plot_axes.plot([1.5, 1.5],[180, 349], linewidth=2, color='white')

cbar = fig.colorbar(pcm, cax=cbar_axes, orientation='vertical')
cbar_axes.text(0.5,1.1,r'$|\hat{u}_x|_{z=1}$',va='center',ha='center',fontsize=fontsize,transform=cbar_axes.transAxes)


#plot_axes.set_xlim([1.5e-2,1.5])
#plot_axes.set_ylim([1e-7, 1e-1])

#plot_axes.text(0.5,1.07,r'$\ell=1$',va='center',ha='center',fontsize=fontsize,transform=plot_axes.transAxes)

plot_axes.set_xlabel(r'$2\pi\, f/N$', fontsize=fontsize)
plot_axes.set_ylabel(r'$t/\tau_c$', fontsize=fontsize)

plot_axes.text(-0.24,1.12,r'$C^{10}$',va='center',ha='left',fontsize=18,transform=plot_axes.transAxes)

plt.savefig('figures/spectrogram.png')

