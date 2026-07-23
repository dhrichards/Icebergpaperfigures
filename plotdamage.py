#%%
import adios4dolfinx
import kraken as kr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
from tqdm import tqdm


filename = f'relaxation_lo_level0.008height300Gc0.5dt2.0psicrit1.0l3.5cellfactor1.0gv_tol3.0_damagemodelAT2higher__.bp'


i = 153

msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=i)
model = kr.base.Simulation(msh, kr.momentum.mixed.SemiLagrangianEpsilon,
                    kr.damage.higherorder.HigherOrder)

adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=i)
adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=i) 

d = kr.plotting.dolfinx_to_array(msh,model.damage.d)

x = msh.geometry.x[:,0]
y = msh.geometry.x[:,1]

cty = kr.plotting.get_connectivity(msh)
tess = tri.Triangulation(
        x, 
        y, 
        triangles=cty)

fig,ax = plt.subplots(figsize=(6,4))
c = ax.tripcolor(tess, d, cmap='viridis', shading='gouraud')

ax.set_aspect('equal')
ax.set_xlim([24.5,26.7])
ax.set_ylim([-1.0,0.1])