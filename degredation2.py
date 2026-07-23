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
    "text.latex.preamble": r"\usepackage{bm}",
})

get_outline = kr.plotting.get_outline
# use a nice ice blue
colorcrack = "#00a6ff"
darkgrey = "#595959"
lightgrey = "#cccccc"

ls = [0.01,0.0025]
degradedornot = ["degraded", "notdegraded"]
 
fig, axs = plt.subplots(2,2, figsize=(8,4))

letters = ['a', 'b', 'c', 'd', 'e', 'f']

#get matlplotib default line colors
linecolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--']
L=5

for ii in range(2):
    
    for j in range(len(ls)):
        l = ls[j]
        # degradationtest_l0.0025_cellfactor1.0_notdegraded.bp
        filename = f'degradationtest_l{l}_cellfactor2.0_{degradedornot[ii]}.bp'

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


        k = 300
        msh.geometry.x[:,0] += k*ux
        msh.geometry.x[:,1] += k*uz

        if ii==0:
            axs[0,ii].plot(*get_outline(msh),label=f'$\ell^*={lstar}$')
            axs[0,ii].plot(*get_outline(msh0),color='gray',label='Iceberg')
        else:
            axs[0,ii].plot(*get_outline(msh))
            axs[0,ii].plot(*get_outline(msh0),color='gray')


        axs[0,ii].set_aspect('equal')
        axs[0,ii].set_xlim(1,L+0.3)
        axs[0,ii].set_ylim(-0.5,2.5)
        axs[0,ii].axis('off')

        tess = kr.plotting.get_triangulation(msh0)



        cmap = cmr.get_sub_cmap('cmr.freeze_r',0,1)

        d = np.clip(d,0,1)

        

        axs[1,ii].tricontour(
        tess,
        d,
        levels=[0.5],
        colors=linecolors[j],
        )

        axs[1,ii].set_aspect('equal')
        axs[1,ii].set_xlim(L-1.5,L+0.3)
        axs[1,ii].set_ylim(0.8,1.1)
        axs[1,ii].axis('off')

     
        

axs[1].set_title(r'(b) $-\nabla \cdot \bm{\widetilde{\sigma}}= \bm{f} $')
axs[0].set_title(r'(a) $-\nabla \cdot \bm{\widetilde{\sigma}}= g \bm{f} $')

#add legend add bottom - only labels from first axis
fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout()

fig.savefig('degradation.pdf', bbox_inches='tight')
fig.savefig('degradation.png', bbox_inches='tight', dpi=600)
