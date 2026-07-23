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

get_outline = kr.plotting.get_outline
# use a nice ice blue
colorcrack = "#00a6ff"
darkgrey = "#595959"
lightgrey = "#cccccc"


filename = f'iceberg_level0.0_height350.0_Kic400.0_dt0.25_sigmacdeg20_sigmac0200_lstar0.075_cellfactor3.0_Ttop20_Tbot20_nondimlength10.0__.bp'

t = adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, function_name = "w_damage")

itstoplot = [1,2,31,-2,-1]
fig, axs = plt.subplots(len(itstoplot), 1, figsize=(8,4.5))
axs = axs.flatten()
dt = 0.25
letters = ['a', 'b', 'c', 'd', 'e', 'f']
j = 0

msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)

for i, ax in zip(itstoplot, axs):
    # if i ==2:
    #     break
    msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=t[i])

    
    model = kr.base.Simulation(msh)
    model.setup()
    model.read_checkpoint(filename, t=t[i])
    

    adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=t[i])
    adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=t[i]) 
    adios4dolfinx.read_function(filename, model.damage.w_prev_it2, name ="w_prev_it2_damage", time=t[i])
    # model.setup()
    if i == -1:
        d = kr.plotting.dolfinx_to_array(msh,model.damage.d_prev_it2)
    else:
        d = kr.plotting.dolfinx_to_array(msh,model.damage.d)

    σ = es.cauchy_stress(model.momentum.ε_e,model.params.ν)
    λ,_ = mf.eigenstate(σ)

    ev = kr.plotting.dolfinx_to_array(msh, 
        model.momentum.ψplus/model.params.ψcritstar)
    
    dx = msh.geometry.x[:,0] - msh0.geometry.x[:,0]
    dy = msh.geometry.x[:,1] - msh0.geometry.x[:,1]
    k = 250
    msh.geometry.x[:,0] = msh0.geometry.x[:,0] + k*dx
    msh.geometry.x[:,1] = msh0.geometry.x[:,1] + k*dy

    # Plot all in one go with a single label
    ax.plot(*get_outline(msh0), lw=0.5, color=lightgrey,label='Initial outline',alpha=0.75)
    ax.plot(*get_outline(msh), lw=0.5,color=darkgrey,label=r'Displacement $\times 250$')
    
    
    

    # cty = kr.plotting.get_connectivity(msh)
    # tess = tri.Triangulation(
    #         msh.geometry.x[:,0], 
    #         msh.geometry.x[:,1],
    #         triangles=cty)
    
    tess = kr.plotting.get_triangulation(msh)

    

    # c = ax.tricontour(tess, d, levels=[0.95], linewidths=1,colors=colorcrack,label='$d=0.5$')
    #log scale
    ev = np.clip(ev, 1e-5, 1)
    ev = np.log10(ev)
    c = ax.tricontourf(tess, ev, levels=10, cmap='Blues', vmin=-5, vmax=0)

    #black filled in contour for damage
    c2 = ax.tricontourf(tess, d, levels=[0.8,1.2], colors='black')

    ax.set_aspect('equal')

    #hide axes
    ax.axis('off')

    # ax.set_xlim([-0.5,10.5])
    ax.set_ylim([-0.15,1.1])
    
    t_days = t[i-1]/(3600*24)
    # ax.set_title(f'$t = {t_days:.1f}$ days',fontsize=11)
    # ax.text(5, 1.1, f'({letters[j]}) $t = {t_days:.2f}$ days',ha ='center', va='bottom',fontsize=11)
    ax.text(-2.3, 0.5, f'({letters[j]}) $t = {t_days:.2f}$ days',ha ='left', va='center',fontsize=11)
    j += 1
# add text for each row showing level
# axs[j,0].text(-0.5, 0.5, f'crack water level \n above sea level = {level*300:.1f} m', transform=axs[j,0].transAxes,
            #   fontsize=11, va='center', ha='left', color='black')

fig.tight_layout()


handles, labels = axs[0].get_legend_handles_labels()

handles = [handles[0], handles[1]]
labels = [labels[0], labels[1]]

handles.append(
    plt.Line2D([0], [0], color='black', lw=4)
)
labels.append(r'$d>0.8$')

fig.subplots_adjust(bottom=0.08)
fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=3,
    
)


cbar = fig.colorbar(c, 
    ax=axs,
    orientation='vertical',
    fraction=0.02,
    pad=0.01
)


cbar.set_label(r'$\psi^+ / \psi_{crit}$', fontsize=11)

cbar.set_ticks([-5, -4, -3, -2, -1, 0])
cbar.set_ticklabels([
    r'$10^{-5}$',
    r'$10^{-4}$',
    r'$10^{-3}$',
    r'$10^{-2}$',
    r'$10^{-1}$',
    r'$1$'
])


fig.savefig('damage_subplots.png', dpi=400, bbox_inches='tight')
fig.savefig('damage_subplots.pdf', bbox_inches='tight')
