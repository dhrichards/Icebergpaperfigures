#%%
import adios4dolfinx
import kraken as kr
import kraken.numerics.energy_splits as es
import kraken.numerics.maths_functions as mf
import numpy as np
import ufl
from matplotlib import cm, pyplot as plt
from matplotlib import tri, gridspec
from mpi4py import MPI
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import cmasher as cmr
#use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "text.latex.preamble": r"\usepackage{bm}",
})


get_outline = kr.plotting.get_outline
# use a nice ice blue
colorcrack = "#00a6ff"
darkgrey = "#595959"
lightgrey = "#cccccc"

ls = [0.01,0.005,0.0025,0.001]
degradedornot = ["degraded", "notdegraded"]

# add gridspec, outline at top showing region, then ls x 2 subplots below
fig = plt.figure(figsize=(8,4))

gs = gridspec.GridSpec(2,1, height_ratios=[1, 3], hspace=0.1)

# split lower part into 2 columns, len(ls) rows
gs_lower = gridspec.GridSpecFromSubplotSpec(len(ls), 2, subplot_spec=gs[1], wspace=0.3, hspace=0.3)

letters = ['a', 'b', 'c', 'd', 'e', 'f']

#get matlplotib default line colors
linecolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--']
L=5

for ii in range(2):

    for j in range(len(ls)):
        l = ls[j]
        # degradationtest_l0.0025_cellfactor1.0_notdegraded.bp
        filename = f'degradationtest_l{l}_cellfactor1.0_{degradedornot[ii]}.bp'

        # extract attributes fromf filename
        L = 5.0
        H = 400
        lstar = float(filename.split('l')[1].split('_')[0])
        cellfactor = float(filename.split('cellfactor')[1].split('_')[0])
        
        t = adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, function_name = "w_damage")

       
       

        # if i ==2:
        msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)
        #     break
        msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=t[1])
        model = kr.base.Simulation(msh)

        x = ufl.SpatialCoordinate(msh)
        z = x[msh.geometry.dim-1]
        model.params.T.value = -5
        model.params.A0.value = mf.rate_factor_np(-5)
        model.params.H.value = H
        model.params.l.value = lstar*H
        model.params.dt.value = 2.5*24*60*60
        model.params.Kic.value = 100*1e3
        model.params.patm.value = 0.0
        model.params.crack_level_above_sea.value = -0.9*H
        model.params.sea_level.value = 0.9*H
        model.params.length.value =  L * H

        model.params.σc.value = 200e3

    
    
        
        model.setup()
        model.read_checkpoint(filename, t=t[1])
        

        adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=t[1])
        adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=t[1]) 
        adios4dolfinx.read_function(filename, model.damage.w_prev_it2, name ="w_prev_it2_damage", time=t[1])



        

        

      
#     # model.setup()
    
        d = kr.plotting.dolfinx_to_array(msh,model.damage.d)
        ux = kr.plotting.dolfinx_to_array(msh,model.momentum.u[0])*model.params.ucstar_float
        uz = kr.plotting.dolfinx_to_array(msh,model.momentum.u[1])*model.params.ucstar_float


        
        ax = plt.subplot(gs_lower[j,ii])
   
      

        tess = kr.plotting.get_triangulation(msh)



        cmap = cmr.get_sub_cmap('cmr.freeze_r',0,1)

        d = np.clip(d,0,1)

        

        c=ax.tricontourf(
        tess,
        d,
        cmap = cmap,
        levels = np.linspace(0,1,5),
        )
        axmin = L-2.1
        axmax = L+0.05
        ax.set_xlim(axmin,axmax)
        axymin = 0.8
        axymax = 1.1
        ax.set_ylim(axymin,axymax)

        ax.set_aspect('equal')
   
        ax.axis('off')

plt.subplot(gs_lower[0,0]).set_title(r'(a) $-\nabla \cdot \bm{\widetilde{\sigma}}= g \bm{f} $')
plt.subplot(gs_lower[0,1]).set_title(r'(b) $-\nabla \cdot \bm{\widetilde{\sigma}}= \bm{f} $')


#add colorbar, lower horizontal

cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
#not continous colorbar

cbar = fig.colorbar(
    c,
    cax=cax,
    orientation='horizontal'
)
cbar.set_label(r'$d$')

for l in ls:
    ax = plt.subplot(gs_lower[ls.index(l),0])
    ax.text(1.135, 0.0, r'$\ell^* = {}$'.format(l), transform=ax.transAxes, ha='center', va='bottom')
ax = plt.subplot(gs[0])
ax.axis('off')
#get colour for d=0 from cmap
c0 = cmap(0.125)
# plot outline and fill with c0, linecolour black
ax.add_patch(plt.Rectangle((0,0),L,1,fill=True,color=c0))
ax.add_patch(plt.Rectangle((0,0),L,1,fill=False,color='k',lw=1.5))

#plot rectangle showing region in zoomed plot
ax.add_patch(plt.Rectangle((axmin,axymin),axmax-axmin,axymax-axymin,fill=False,color='gray',lw=1.5))

ax.set_aspect('equal')
ax.set_xlim(-0.1,L+0.3)
ax.set_ylim(-0.1,1.2)



fig.tight_layout()

fig.savefig('degradationcontour.pdf', bbox_inches='tight')
fig.savefig('degradationcontour.png', bbox_inches='tight', dpi=600)
