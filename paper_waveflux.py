
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

t_mar, b_mar, l_mar, r_mar = (0.26, 0.4, 0.6, 0.12)
h_plot, w_plot = (1., 1./publication_settings.golden_mean)
h_pad = 0.5
w_pad = 0.1

h_total = t_mar +   h_pad + 2*h_plot + b_mar
w_total = l_mar + 2*w_pad + 3*w_plot + r_mar

width = 7.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# plots
plot_axes = []
for i in range(3):
  for j in range(2):
    left = (l_mar + i*w_pad + i*w_plot) / w_total
    bottom = 1 - (t_mar + j*h_pad + (j+1)*h_plot ) / h_total
    width = w_plot / w_total
    height = h_plot / h_total
    plot_axes.append(fig.add_axes([left, bottom, width, height]))

# load data
files = ['Ra2e8_damping/slices/slices_c_freq.h5','Ra1e9_damping/slices/slices_c_freq.h5','Ra1e10_damping/slices/slices_c_freq.h5']
N_list = [np.sqrt(100*60/0.5)/(2*np.pi),np.sqrt(100*100/0.5)/(2*np.pi),np.sqrt(100*200/0.5)/(2*np.pi)]
a_list = [3, 3, 3]
b_list = [-15/2, -13/2, -13/2]
amp_list = [1e-5, 2e-6, 2e-5]

sim_index = int(sys.argv[1])

file = files[sim_index]
amp = amp_list[sim_index]
a = a_list[sim_index]
b = b_list[sim_index]

f = h5py.File(file)

freq = np.array(f['scales/f'])
kx = np.array(f['scales/kx'])

w = np.array(f['tasks/w z=0.6'])
p = np.array(f['tasks/p z=0.6'])
flux = np.real(w * np.conj(p))

f.close()

# add positive and negative frequency
nf = flux.shape[0]
mid_f = int((nf-1)/2)
if nf % 2 == 0:
    flux = flux[1:mid_f+1] + flux[-1:mid_f+1:-1]
    freq = freq[1:mid_f+1]
else:
    flux = flux[1:mid_f+1] + flux[-1:mid_f:-1]
    freq = freq[1:mid_f+1]

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

#colors = ['MidnightBlue', 'Firebrick', 'DarkGoldenrod']
colors = ['MidnightBlue']*3
ratio = 0.5
#light_colors = [change_brightness(midnight_blue, ratio), change_brightness(firebrick_red, ratio), change_brightness(dark_goldenrod, ratio)]
light_colors = [change_brightness(firebrick_red, ratio)]*3

kx_target = [2, 5, 10]
f_target = [0.2, 0.5, 0.8]

for i in range(3):
  for j in range(2):
    ax = plot_axes[2*i+j]

    for k in range(1):
      N = N_list[sim_index]

      if j == 0:
        ax.loglog(freq/N, flux[:, kx_target[i]], color=colors[i])
        #prediction = amp * (tau/2/np.pi) * (kx_target[i])**a * (freq*tau)**b
        prediction = amp * (kx_target[i])**a * freq**b
        ax.loglog(freq/N, prediction, color=light_colors[i])

        ax.set_xlim([2e-2,1.5])
        ax.set_ylim([1e-15,1e-5])

      if j == 1:
        i_f = np.argmin(np.abs(freq/N-f_target[i]))
        ax.loglog(kx[1:]/(2*np.pi), flux[i_f, 1:], color=colors[i])
        #prediction = amp * (tau/2/np.pi) * (kx[1:]/2/np.pi)**a * (freq[i_f]*tau)**b
        prediction = amp * (kx[1:]/2/np.pi)**a * freq[i_f]**b
        ax.loglog(kx[1:]/(2*np.pi), prediction, color=light_colors[i])

        ax.set_xlim([0.8, 100])
        flux_max = np.max(flux[i_f, 1:])
        ax.set_ylim([1e-17,1e-6])

# plot axis labels
for i in range(3):
  for j in  range(2):
    if i > 0:
      ax = plot_axes[2*i+j]
      plt.setp(ax.get_yticklabels(), visible=False)

text = [r'$\ell=2$',r'$\dfrac{2\pi\, f}{N}=0.2$',r'$\ell=5$',r'$\dfrac{2\pi\, f}{N}=0.5$',r'$\ell=10$',r'$\dfrac{2\pi\, f}{N}=0.8$']
for i in range(3):
  for j in range(2):
    ax = plot_axes[2*i+j]
    if j == 0:
      ax.text(0.95,0.83,text[2*i+j],va='center',ha='right',fontsize=fontsize,transform=ax.transAxes)
    elif j == 1:
#      ax.text(0.95,0.18,text[2*i+j],va='center',ha='right',fontsize=fontsize,transform=ax.transAxes)
      if i < 2:
        ax.text(0.05,0.18,text[2*i+j],va='center',ha='left',fontsize=fontsize,transform=ax.transAxes)
      else:
        ax.text(0.05,0.83,text[2*i+j],va='center',ha='left',fontsize=fontsize,transform=ax.transAxes)


class CustomLogFormatter(LogFormatterMathtext):
  
    def __call__(self, x, pos = None):
  
        if x in [1, 10]:
            return LogFormatterMathtext.__call__(self, x, pos = None)
  
        else:
            return ' '


for i in range(3):
  for j in range(2):
    ax = plot_axes[2*i+j]
    if j == 1:
      if i < 2:
        ax.xaxis.set_major_formatter(CustomLogFormatter())

for i in range(3):
  for j in range(2):
    ax = plot_axes[2*i+j]
    if j == 0:
      ax.set_xlabel(r'$2\pi\, f/N$', fontsize=fontsize)
    elif j == 1:
      ax.set_xlabel(r'$\ell$', fontsize=fontsize)

for i in range(3):
  for j in range(2):
    ax = plot_axes[2*i+j]
    if i == 0:
      ax.set_ylabel(r'$\delta F$', fontsize=fontsize)

ax = plot_axes[0]
sim_list = [r'$D^8$', r'$D^9$', r'$D^{10}$']
ax.text(-0.24,1.12,sim_list[sim_index],va='center',ha='left',fontsize=18,transform=ax.transAxes)

name_list = ['D8','D9','D10']

plt.savefig('figures/waveflux_%s.eps' %name_list[sim_index])
#plt.savefig('figures/waveflux_%s.png' %name_list[sim_index])

