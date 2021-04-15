
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import h5py
import publication_settings
import sys

matplotlib.rcParams.update(publication_settings.params)

fontsize = 12

t_mar, b_mar, l_mar, r_mar = (0.24, 0.4, 0.45, 0.1)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)
h_pad = 0.1
w_pad = 0.1

h_total = t_mar +   h_pad + 2*h_plot + b_mar
w_total = l_mar +             w_plot + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
plot_axes = []
for i in range(2):
    left = (l_mar) / w_total
    bottom = 1 - (t_mar + i*h_pad + (i+1)*h_plot ) / h_total
    width = w_plot / w_total
    height = h_plot / h_total
    plot_axes.append(fig.add_axes([left, bottom, width, height]))

# load data
files = ['Ra1e10/slices/slices_c_freq.h5','Ra1e10/slices_rst2/slices_rst2_c_freq.h5']
N = np.sqrt(100*200/0.5)/(2*np.pi)

freq_list = []
ux_list = []

for file in files:

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

    ux2 = np.sum(ux2[:, 1:], axis=1)

    freq_list.append(freq)
    ux_list.append(np.sqrt(ux2))

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

color = 'MidnightBlue'

for i in range(2):
    ax = plot_axes[i]
    ax.loglog(freq_list[i]/N, ux_list[i], color=color)

    ax.set_xlim([1.5e-2,1.5])
    ax.set_ylim([1e-6, 1e-1])

# plot axis labels
plt.setp(plot_axes[0].get_xticklabels(), visible=False)

t_start = []
t_start = [9, 180]
t_end = [80, 349]

for i in range(2):
    ax = plot_axes[i]
#    ax.text(0.5,1.08,r'$%i \tau_c < t < %i \tau_c$' %(t_start[i], t_end[i]),va='center',ha='center',fontsize=fontsize,transform=ax.transAxes)
#    ax.text(0.5,0.1,r'$%i \tau_c < t < %i \tau_c$' %(t_start[i], t_end[i]),va='center',ha='center',fontsize=fontsize,transform=ax.transAxes)
    ax.text(0.05,0.1,r'$%i \tau_c < t < %i \tau_c$' %(t_start[i], t_end[i]),va='center',ha='left',fontsize=fontsize,transform=ax.transAxes)

plot_axes[1].set_xlabel(r'$f/N$', fontsize=fontsize)

for i in range(2):
    ax = plot_axes[i]
    ax.set_ylabel(r'$\left|\hat{u}_x\right|_{z=L}$', fontsize=fontsize)


ax = plot_axes[0]
ax.text(-0.24,1.12,r'$C^{10}$',va='center',ha='left',fontsize=18,transform=ax.transAxes)

plt.savefig('figures/spectrum_C10.eps')
#plt.savefig('figures/spectrum_C10.png')

