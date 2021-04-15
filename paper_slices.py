
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from scipy.optimize import brentq
import publication_settings
from dedalus.extras import plot_tools
import brewer2mpl
import dedalus.public as de

matplotlib.rcParams.update(publication_settings.params)

color_map = ('RdBu', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap

color_map = ('PuOr', 'diverging', 11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap2 = b2m.mpl_colormap

cmaps = [cmap1,cmap2]

fontsize = 14

t_mar, b_mar, l_mar, r_mar = (0.3, 0.3, 0.3, 0.25)
h_slice, w_slice = (1., 1.)
h_pad = 0.07

h_cbar_top, w_cbar_top = (0.05, w_slice)
h_cb_pad = 0.05

h_cbar_right, w_cbar_right = (h_slice, 0.05)
w_pad = 0.05

h_total = t_mar + 2*h_pad + h_cb_pad + h_cbar_top + 3*h_slice + b_mar
w_total = l_mar + w_pad + w_cbar_right + 1*w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(3):
  left = (l_mar) / w_total
  bottom = 1 - (t_mar + h_cbar_top + h_cb_pad + i*h_pad + (i+1)*h_slice ) / h_total
  width = w_slice / w_total
  height = h_slice / h_total
  slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_cbar_top) / h_total
width = w_cbar_top / w_total
height = h_cbar_top / h_total
cbar_top = fig.add_axes([left, bottom, width, height])

cbar_right = []
for i in range(3):
  left = (l_mar + w_slice + w_pad) / w_total
  bottom = 1 - (t_mar + h_cbar_top + h_cb_pad + i*h_pad + (i+1)*h_slice ) / h_total
  width = w_cbar_right / w_total
  height = h_cbar_right / h_total
  cbar_right.append(fig.add_axes([left, bottom, width, height]))

# load slice data
T_list = []
u_list = []
w_list = []
x_list = []
y_list = []

file_num = np.array([24, 17, 11])
output_num = np.array([-5, -5, -5])
Ra_list = ['2e8','1e9','1e10']

for i in range(3):
  f = h5py.File('Ra%s_damping/snapshots/snapshots_s%i.h5' %(Ra_list[i], file_num[i]))
  T_list.append(np.array(f['tasks/T'][output_num[i],:,:]))
  u_list.append(np.array(f['tasks/u'][output_num[i],:,:]))
  w_list.append(np.array(f['tasks/w'][output_num[i],:,:]))

  x_list.append(np.array(f['scales/x/1.0']))
  y_list.append(np.array(f['scales/z/1.0']))

  f.close()

lw = 1

# plot slices
c_im = []

for i in range(3):

  x = x_list[i]
  y = y_list[i]
  xm, ym = plot_tools.quad_mesh(x, y)

  T = np.copy(T_list[i])

  T[T < 0] += np.nan

  c_im.append(slice_axes[i].pcolormesh(xm, ym, T.T, cmap=cmaps[0]))

  T = np.copy(T_list[i])
  u = u_list[i]
  w = w_list[i]
  om = np.gradient(w, x, y)[0] - np.gradient(u, x, y)[1]

  om[T > 0] += np.nan

  c_im.append(slice_axes[i].pcolormesh(xm, ym, om.T, cmap=cmaps[1]))

for slice_axis in slice_axes:
  slice_axis.axis([-0.5,0.5,0,1])

om_lims = [1.5, 2, 4]

for i in range(3):
  c_im[2*i].set_clim(0, 1)
  om_lim = om_lims[i]
  c_im[2*i+1].set_clim([-om_lim,om_lim])

# slice axis labels
for i in range(2):
  plt.setp(slice_axes[i].get_xticklabels(), visible=False)

sim_names = [r'$D^8$',r'$D^9$',r'$D^{10}$']
for i in range(3):
  slice_axes[i].text(0.05,0.918,sim_names[i],va='center',ha='left',fontsize=fontsize,transform=slice_axes[i].transAxes)

for i in range(3):
  if i == 0:
      slice_axes[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
  else:
      slice_axes[i].yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))

for slice_axis in slice_axes:
  slice_axis.set_ylabel(r'$z/L$', fontsize=fontsize)

slice_axes[-1].set_xlabel(r'$x/L$', fontsize=fontsize)

# colorbar
cbar_top_axes = fig.colorbar(c_im[0], cax=cbar_top, orientation='horizontal', ticks=[0, 0.5, 1])
cbar_top.text(0.5,4.5,r'$T$',va='center',ha='center',fontsize=fontsize,transform=cbar_top.transAxes)
cbar_top.xaxis.set_ticks_position('top')

cbars = []
for i in range(3):
  cbars.append( fig.colorbar(c_im[2*i+1], cax=cbar_right[i], orientation='vertical', ticks=MaxNLocator(nbins=3)) )
  cbar_right[i].text(3,1,r'$\omega$',va='center',ha='center',fontsize=fontsize,transform=cbar_right[i].transAxes)

plt.savefig('figures/convection.png',dpi=600)

