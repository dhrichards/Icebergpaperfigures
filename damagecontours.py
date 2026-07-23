#%%
import adios4dolfinx
import kraken as kr
import kraken.numerics.maths_functions as mf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
from tqdm import tqdm
import cmasher as cmr
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

filename = f'ssa_H600.0_l0.01_cellfactor2.0__.bp'
# use a nice ice blue
colorcrack = "#00a6ff"
get_outline = kr.plotting.get_outline

H = float(filename.split('H')[1].split('_')[0])
lstar = float(filename.split('l')[1].split('_')[0])
cellfactor = float(filename.split('cellfactor')[1].split('_')[0])



ztensions = np.array([0.44944945, 0.94994995])
times = adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, function_name = "w_damage")
fig,ax = plt.subplots(figsize=(4,3))



msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=times[1])


model = kr.base.Simulation(msh)

model.params.T.value
model.params.A0.value = mf.rate_factor_np(model.params.T.value)
model.params.H.value = H
model.params.l.value = lstar*H
model.params.dt.value = 2.5*24*60*60
model.params.Kic.value = 100*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = 0.0
model.params.sea_level.value = 0.9*H
model.params.length.value =  2*H


model.setup()
model.read_checkpoint(filename, t=times[1])
d = kr.plotting.dolfinx_to_array(msh,model.damage.d)



tess = kr.plotting.get_triangulation(msh)


cmap = cmr.get_sub_cmap('cmr.freeze_r',0,1)
# cmap = sns.cubehelix_palette(start=, rot=0, dark=0, light=1, as_cmap=True)
c = ax.tricontourf(tess, d, levels=10, cmap=cmap,shading='gouraud')

ax.set_aspect('equal')

ax.axhline(ztensions[0], ls='--', color='k', label = 'Tension threshold')
ax.axhline(ztensions[1], ls='--', color='k')

ax.axhline(0.0, color='k', label = 'Outline')
ax.axhline(1.0, color='k')
ax.axis('off')

# Leave room at bottom
fig.tight_layout(rect=[0, 0.1, 1, 1])

# Common geometry
w = 0.38
h = 0.04
y = 0.03

# -------------------------
# Legend (bottom left)
# -------------------------
leg_ax = fig.add_axes([0.08, y, w, h])
leg_ax.axis('off')

handles, labels = ax.get_legend_handles_labels()

leg_ax.legend(
    handles,
    labels,
    loc='center',
    ncol=1,
    frameon=False,
    fontsize=10,
    handlelength=2.0
)

# -------------------------
# Colorbar (bottom right)
# -------------------------
cax = fig.add_axes([0.54, y, w, h])

cbar = fig.colorbar(
    c,
    cax=cax,
    orientation='horizontal'
)

cbar.set_label(r'$d$', fontsize=11)

fig.savefig('ssa.png', dpi=400, bbox_inches='tight')
fig.savefig('ssa.pdf', bbox_inches='tight')
