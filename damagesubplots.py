#%%
import adios4dolfinx
import kraken as kr
import kraken.numerics.energy_splits as es
import kraken.numerics.maths_functions as mf
import numpy as np
import ufl
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
from tqdm import tqdm
from matplotlib.colors import ListedColormap

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


filename = f'icebergsymm_L5.0_H500.0_l0.005_dt2.5_relaxt400.0_sigmacdeg0.0_sigmac0200_level0.0_Kic100_cellfactor1.0_Ttop5.0_Tbot5.0_lfactor2.0__.bp'

# extract attributes fromf filename
L = float(filename.split('L')[1].split('_')[0])
H = float(filename.split('H')[1].split('_')[0])
lstar = float(filename.split('l')[1].split('_')[0])
dt = float(filename.split('dt')[1].split('_')[0])
strength_deg = float(filename.split('sigmacdeg')[1].split('_')[0])
strength0 = float(filename.split('sigmac0')[1].split('_')[0])
level = float(filename.split('level')[1].split('_')[0])
Kic = float(filename.split('Kic')[1].split('_')[0])
cellfactor = float(filename.split('cellfactor')[1].split('_')[0])
Ttop = float(filename.split('Ttop')[1].split('_')[0])
Tbot = float(filename.split('Tbot')[1].split('_')[0])
nondim_length = float(filename.split('lfactor')[1].split('_')[0])

t = adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, function_name = "w_damage")

itstoplot = [0,1,len(t)//3,2*len(t)//3,-2, -1]
fig, axs = plt.subplots(2, len(itstoplot)//2, figsize=(8,4.5))
axs = axs.flatten()
letters = ['a', 'b', 'c', 'd', 'e', 'f']
j = 0

msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)


for i, ax in zip(itstoplot, axs):
    # if i ==2:
    #     break
    msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=t[i])
    model = kr.base.Simulation(msh)

    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    model.params.T = Tbot + (Ttop - Tbot)*z
    model.params.A0.value = mf.rate_factor_np(Ttop)
    model.params.H.value = H
    model.params.l.value = lstar*H
    model.params.dt.value = dt*24*60*60
    model.params.Kic.value = Kic*1e3
    model.params.patm.value = 0.0
    model.params.crack_level_above_sea.value = level
    model.params.sea_level.value = 0.9*H
    model.params.length.value =  L * H

    model.params.σc = strength0*1e3 - strength_deg*1e3*(model.params.T)

   
    
    
    model.setup()
    model.read_checkpoint(filename, t=t[i])
    

    adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=t[i])
    adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=t[i]) 
    adios4dolfinx.read_function(filename, model.damage.w_prev_it2, name ="w_prev_it2_damage", time=t[i])

    # model.setup()
    
    d = kr.plotting.dolfinx_to_array(msh,model.damage.d)

    σ = es.cauchy_stress(model.momentum.ε_e,model.params.ν)
    λ,_ = mf.eigenstate(σ)

    ev = kr.plotting.dolfinx_to_array(msh, 
        model.momentum.ψplus/model.params.ψcritstar)
    
    dx = msh.geometry.x[:,0] - msh0.geometry.x[:,0]
    dy = msh.geometry.x[:,1] - msh0.geometry.x[:,1]
    k = 1
    msh.geometry.x[:,0] = msh0.geometry.x[:,0] + k*dx
    msh.geometry.x[:,1] = msh0.geometry.x[:,1] + k*dy

    # Plot all in one go with a single label
    ax.plot(*get_outline(msh0), lw=0.5, color=lightgrey,label='Initial outline',alpha=0.75)
    ax.plot(*get_outline(msh), lw=0.5,color=darkgrey,label=r'Displacement')
    
    
    

    # cty = kr.plotting.get_connectivity(msh)
    # tess = tri.Triangulation(
    #         msh.geometry.x[:,0], 
    #         msh.geometry.x[:,1],
    #         triangles=cty)
    
    tess = kr.plotting.get_triangulation(msh)

    

    # c = ax.tricontour(tess, d, levels=[0.95], linewidths=1,colors=colorcrack,label='$d=0.5$')
    #log scale
    ev = np.clip(ev, 1e-5, 100)
    ev = np.log10(ev)

    blues = plt.cm.Blues(np.linspace(0.2, 1, 10))
    cmap = ListedColormap(blues)

    # colour for values above vmax
    cmap.set_over('red')
    c = ax.tricontourf(tess, 
                       ev, 
                       levels=np.linspace(-5,0,11),
                        cmap=cmap, vmin=-5, vmax=0, extend='max')
    for artist in c.get_children():
        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor("face")
            artist.set_linewidth(0)
    

    #black filled in contour for damage
    d_val = 0.75
    c2 = ax.tricontourf(tess, d, levels=[d_val,1.2], colors='black')

    ax.set_aspect('equal')

    #hide axes
    ax.axis('off')

    ax.set_xlim([L-1,L+0.12])
    ax.set_ylim([-0.01,1.01])
    
    t_days = t[i]/(3600*24)
    ax.set_title(f'({letters[j]}) $t = {t_days:.1f}$ days',fontsize=11)
    # ax.text(5, 1.1, f'({letters[j]}) $t = {t_days:.2f}$ days',ha ='center', va='bottom',fontsize=11)
    # ax.text(-2.3, 0.5, f'({letters[j]}) $t = {t_days:.2f}$ days',ha ='left', va='center',fontsize=11)
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
labels.append(fr'$d>${d_val:.2f}')

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
    pad=0.01,
    extend='max'
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
