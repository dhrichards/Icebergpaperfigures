#%%
import adios4dolfinx
import kraken as kr
import kraken.numerics.energy_splits as es
import kraken.numerics.maths_functions as mf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
from tqdm import tqdm
#use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# use a nice ice blue
colorcrack = "#00a6ff"



filename = f'relaxation_lo_level0.0height300.0Gc0.5dt1.0psicrit1.0l20.0cellfactor1.0gv_tol3.0_damagemodelAT2higher__.bp'
fig, ax = plt.subplots(figsize=(7,3))

msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)

times =adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, "w_momentum")

# time = time - time[10]

i = 9
# for ax,i in zip(axs.flatten(),istoplot):


msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=times[i])


model = kr.base.Simulation(msh, kr.momentum.mixed.SemiLagrangianEpsilon,
                    kr.damage.higherorder.HigherOrder)
adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=times[i])
adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=times[i]) 
adios4dolfinx.read_function(filename, model.momentum.ε_e_prev_time, name ="epsiloneprevtime", time=times[i])

σ = es.cauchy_stress(model.momentum.ε_e,model.params.ν)
λ,_ = mf.eigenstate(σ)

ev = kr.plotting.dolfinx_to_array(msh, 
    es.free_energy_plus_lo(model.momentum.ε_e, model.params.ν))
ev = kr.plotting.dolfinx_to_array(msh, λ[1])

dx = msh.geometry.x[:,0] - msh0.geometry.x[:,0]
dy = msh.geometry.x[:,1] - msh0.geometry.x[:,1]

k = 100
msh.geometry.x[:,0] = msh0.geometry.x[:,0] + k*dx
msh.geometry.x[:,1] = msh0.geometry.x[:,1] + k*dy



tess = kr.plotting.get_triangulation(msh)


# ax.plot(*kr.plotting.get_outline(msh), lw=1.0,color='black',label='Iceberg outline')
ax.plot(*kr.plotting.get_outline(msh0), lw=0.5, ls='--', color='gray',label='Initial outline')

# ax.set_aspect('equal')
ax.axis('off')
# Plot all in one go with a single label
# ax.plot(*kr.plotting.get_outline(msh), lw=1.0,color='black',label='Iceberg outline')
# ax.plot(*kr.plotting.get_outline(msh0), lw=0.5, ls='--', color='gray',label='Initial outline')

# Plot waterlines
# ax.hlines(0, xmin=0, xmax=27.0, colors='blue', linestyles='dotted', lw=1.0, label='Sea level')
# level = 0.01
# ax.hlines(level, xmin=0, xmax=27.0, colors='purple', linestyles='dotted', lw=1.0,label =\
#             r'Water level in cracks (+ {l} m)'.format(l=level*300))

# msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=times[i])
# tess = kr.plotting.get_triangulation(msh)

# ax = axs[1]
c2 = ax.tripcolor(tess, ev, shading='gouraud', cmap='viridis')
ax.tricontour(tess,ev, levels=20, colors='black', linewidths=0.5, alpha=0.5)
# # fig.colorbar(c2, ax=ax, label='Max. principal stress (Pa)')

# ax.set_aspect('equal')
# # colorbar
# # fig.colorbar(c2, ax=ax)


# ax.set_xlim([24.5,27])
# ax.set_ylim([-1.0,0.2])

# #hide axes
# ax.axis('off')

# ax.set_title(f't = {time[i]:.0f} days')

#make legend 
handles, labels = ax.get_legend_handles_labels()
# add extra handle for damage contour
# handles.append(plt.Line2D([0], [0], color=colorcrack, lw=1.0))
# labels.append('$d=0.5$')
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
# fig.legend([c.collections[0]], ['Damage contour d=0.5'], loc='lower center', ncol=1)
fig.tight_layout()
