#%%
import adios4dolfinx
import kraken as kr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
from tqdm import tqdm

levels = [0.006,0.008,0.01,0.011, 0.012, 0.013,0.014]
dts = [1.0,2.0,2.0,2.0,2.0,2.0,0.1]
#%%
nt_max = 580

times = np.zeros((len(levels),nt_max))
dt0 = 10
for j in range(len(levels)):
    for i in range(10):
        times[j,i] = (i+1)*dt0
    for i in range(10,nt_max):
        times[j,i] = times[j,i-1] + dts[j]


crackdepths = np.zeros_like(times)
crackdepths_without_advection = np.zeros_like(times)

deepest_crack_x = np.zeros_like(times)

for j in range(len(levels)):
    level = levels[j]

    filename = f'relaxation_lo_level{level}height300Gc0.5dt{dts[j]}psicrit1.0l3.5cellfactor1.0gv_tol3.0_damagemodelAT2higher__.bp'

    msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)
    x0 = msh0.geometry.x[:,0]
    y0 = msh0.geometry.x[:,1]

    # cty0 = kr.plotting.get_connectivity(msh0)
    # tess0 = tri.Triangulation(
    #         x0, 
    #         y0, 
    #         triangles=cty0)

    tess0 = kr.plotting.get_triangulation(msh0)

    for i in tqdm(range(nt_max)):
        try:
            msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=i)
        except:
            # move onto next level if time step not found
            break
        model = kr.base.Simulation(msh, kr.momentum.mixed.SemiLagrangianEpsilon,
                            kr.damage.higherorder.HigherOrder)

    # create simulation object to easily make correct function spaces

        try:
            adios4dolfinx.read_function(filename, model.momentum.w, name ="w_momentum", time=i)
            adios4dolfinx.read_function(filename, model.damage.w, name ="w_damage", time=i)
        except:
            break
        
        d = kr.plotting.dolfinx_to_array(msh,model.damage.d)

        x = msh.geometry.x[:,0]
        y = msh.geometry.x[:,1]

   
        # cty = kr.plotting.get_connectivity(msh)


        
        # tess = tri.Triangulation(
        #         msh.geometry.x[:,0], 
        #         msh.geometry.x[:,1], 
        #         triangles=cty)
        tess = kr.plotting.get_triangulation(msh)

        # c = ax.tricontour(tess, d, levels=[0.5], linewidths=0.5)
    # c = ax.tripcolor(tess, d, shading='gouraud', cmap='viridis')
    # fig.colorbar(c, ax=ax, label='Damage d')

    # Plot all in one go with a single label
    # ax.plot(X, Y, lw=1.0)

    ##extract lowest point d=0.5 
        y_crack = y[d >= 0.5]
        y0_crack = y0[d >= 0.5]
        if len(y_crack) > 0:
            crackdepths[j, i] = np.min(y_crack)
            crackdepths_without_advection[j, i] = np.min(y0_crack)
            min_ind = np.argmin(y_crack)
            deepest_crack_x[j, i] = x0[d >= 0.5][min_ind]
        else:
            crackdepths[j, i] = 0.0
            crackdepths_without_advection[j, i] = 0.0
            deepest_crack_x[j, i] = 0.0
#%%
# save data
np.savez('crack_depths_vs_time.npz', levels=levels, times=times, crackdepths=crackdepths, crackdepths_without_advection=crackdepths_without_advection, deepest_crack_x=deepest_crack_x)

#%%

# load data
data = np.load('crack_depths_vs_time.npz')
levels = data['levels']
times = data['times']
crackdepths = data['crackdepths']   
crackdepths_without_advection = data['crackdepths_without_advection']
deepest_crack_x = data['deepest_crack_x']


#%%

times = times - times[:,10][:, np.newaxis]
time2crack = np.zeros(len(levels))
fracture_depth = np.zeros(len(levels))
rate_of_crack_growth = np.zeros(len(levels))
distance2front = np.zeros(len(levels))
crackinds = np.zeros((len(levels)))



for j in range(len(levels)):
    crackind = np.where(crackdepths[j,:]<-0.5)[0]
    crackinds[j] = crackind[0]
    time2crack[j] = times[j,crackind[0]]
    fracture_depth[j] = crackdepths[j,crackind[0]-1]
    rate_of_crack_growth[j] = (fracture_depth[j]-crackdepths[j,10])/time2crack[j]
    # distance2front[j] = np.abs(deepest_crack_x[j,crackind[0]] - np.max(x0))

## set crackdepths = 0 to nan
crackdepths[crackdepths==0] = np.nan
crackdepths_without_advection[crackdepths_without_advection==0] = np.nan

fig,ax = plt.subplots(1,1,figsize=(6,5))
for j in range(len(levels)):
    ax.plot(times[j,10:], crackdepths[j,10:], marker='o', label=f'Level={levels[j]}')
    # ax.plot(times[j,10:], crackdepths_without_advection[j,10:], marker='x', label=f'Level={levels[j]} without advection',ls='--',linewidth=0.5)
ax.legend()
ax.set_xlabel('Time (days)')
ax.set_ylabel('Crack depth')
ax.set_title('Crack depth over time')
ax.set_ylim([-0.5,-0.01])
fig,axs = plt.subplots(1,3,figsize=(8,2))



axs[0].plot(levels, time2crack, marker='o')
axs[0].set_xlabel('Level')
axs[0].set_ylabel('Time for full fracture (days)')
# axs[0].set_yscale('log')
# axs[0].set_xscale('log')


axs[1].plot(levels, -fracture_depth, marker='o')
axs[1].set_xlabel('Level')
axs[1].set_ylabel('Fracture depth before full fracture (m)')
# axs[1].set_aspect('equal')

# axs[2].plot(levels, -rate_of_crack_growth, marker='o')
# axs[2].set_xlabel('Level')
# axs[2].set_ylabel('Rate of crack growth (m/day)')

axs[2].plot(levels, distance2front, marker='o')
axs[2].set_xlabel('Level')
axs[2].set_ylabel('Distance to front at full fracture /H')
axs[2].set_ylim([0, 1 ])
plt.tight_layout()

#%%
fig,ax = plt.subplots(1,1,figsize=(3,3))

levelsm = [300*lev for lev in levels]
ax.plot(levelsm, time2crack, marker='o', linewidth=1.5)
ax.set_xlabel('Water level in crack \n above sea level (m)')
ax.set_ylabel('Calving time (days)')
# ax.grid()
fig.tight_layout()
fig.savefig('calvingtime.png', bbox_inches='tight',dpi=800)
