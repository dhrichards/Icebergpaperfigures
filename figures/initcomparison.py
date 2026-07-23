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
import cmasher as cmr
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
oceanblue = "#000dff"
lightgrey = "#cccccc"


filename1 = f'icebergsymm_L5.0_H500.0_l0.005_dt2.5_relaxt400.0_sigmacdeg0.0_sigmac0200_level0.0_Kic100_cellfactor1.0_Ttop5.0_Tbot5.0_lfactor2.0__.bp'
filename2 = f'iceshelf_L5.0_H500.0_l0.005_dt2.5_relaxt400.0_sigmacdeg0.0_sigmac0200_level0.0_Kic100_cellfactor1.0_Ttop5.0_Tbot5.0_lfactor2.0_noinit_.bp'
filenames = [filename1, filename2]

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(8,4))

gs = GridSpec(
    3, 1,
    height_ratios=[1.2, 1, 1],
    hspace=0.05
)

ax_global = fig.add_subplot(gs[0])
axs = [
    fig.add_subplot(gs[1]),
    fig.add_subplot(gs[2]),
]
j = 0
letters = ['a', 'b', 'c', 'd', 'e', 'f']
for j, (filename, ax) in enumerate(zip(filenames, axs)):

    # extract attributes fromf filename
    L = float(filename.split('L')[1].split('_')[0])
    H = float(filename.split('H')[1].split('_')[0])
    lstar = float(filename.split('_l')[1].split('_')[0])
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

    
    

    msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)

    i = 1
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

    
    X,Y = get_outline(msh)
    
    ind = np.nanargmin(np.abs(Y-0.9))

    
      
        
    # ax.plot([0,X[ind]],[0.9,0.9], lw=0.5, color='k', linestyle = '--', label='Sea level')
        

    
    
    
    cmap = cmr.get_sub_cmap('cmr.freeze_r',0,1)

    d = np.clip(d,0,1)
    # cmap = sns.cubehelix_palette(start=, rot=0, dark=0, light=1, as_cmap=True)
    
    tess = kr.plotting.get_triangulation(msh)

    # ----------------------------------
    # Global panel (only once)
    # ----------------------------------
    if j == 0:

        # c = ax_global.tricontourf(
        #     tess,
        #     d,
        #     levels=10,
        #     cmap=cmap
        # )

        ax_global.plot(
            *get_outline(msh),
            lw=0.5,
            color=darkgrey
        )

        ax_global.set_aspect("equal")
        ax_global.axis("off")

        ax_global.set_xlim([0.1, L+0.5])
        ax_global.set_ylim([-0.1, 1.1])

        ax_global.set_title(r"(a) Ice shelf")

        ax_global.plot([X[ind],10],[0.9,0.9], color=oceanblue, label='Sea level')


    # ----------------------------------
    # Zoomed panel
    # ----------------------------------
    ax.tricontourf(
        tess,
        d,
        levels=10,
        cmap=cmap
    )

    

    ax.plot(
        *get_outline(msh),
        lw=0.5,
        color=darkgrey
    )

    ax.set_aspect("equal")
    ax.axis("off")

    zoom_xlim = [L-2, L+0.1]
    zoom_ylim = [0.8, 1.05]

    ax.set_xlim(zoom_xlim)
    ax.set_ylim(zoom_ylim)

    # rectangle on global panel
    rect = Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        fill=False,
        edgecolor=f"C{j}",
        lw=1.5,
    )

    ax_global.add_patch(rect)

    ax.set_aspect('equal')

    #hide axes
    ax.axis('off')

    ax.set_xlim([L-2,L+0.1])
    ax.set_ylim([0.85,1.01])

    
axs[0].set_title(r"(b) Seeded crevasses")
axs[1].set_title(r"(c) Unseeded crevasses") 
fig.tight_layout()


handles, labels = axs[0].get_legend_handles_labels()

# handles = [handles[0], handles[1]]
# labels = [labels[0], labels[1]]



# fig.legend(
#     handles,
#     labels,
#     loc='lower center',
#     ncol=4,
    
# )


cbar = fig.colorbar(c, 
    ax=axs,
    orientation='vertical',
    fraction=0.02,
    pad=0.01,
    extend='both'
)


cbar.set_label(r'$d$', fontsize=11)


fig.savefig('damage_subplots.png', dpi=400, bbox_inches='tight')
fig.savefig('damage_subplots.pdf', bbox_inches='tight')
