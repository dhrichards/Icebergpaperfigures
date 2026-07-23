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



# filename = f'relaxation_lo_level0.0height300.0Gc0.5dt0.3psicrit1.0l40.0cellfactor2.0gv_tol3.0_damagemodelAT2higher__.bp'
filename = f'relaxation_none_level0.0height300.0Gc0.5dt20.0psicrit1.0l20.0cellfactor1.0gv_tol4.0_damagemodelAT2higher_te_.bp'
# fig, ax = plt.subplots(figsize=(7,3))

msh0 = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=0)

times =adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, "w_momentum")

# time = time - time[10]

i = 3
# for ax,i in zip(axs.flatten(),istoplot):


msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=times[i])


model = kr.base.Simulation(msh, kr.momentum.totalelastic.SemiLagrangian,
                    kr.damage.higherorder.HigherOrder)

model.read_checkpoint(filename, t=times[i])


import ufl

n = 3.0


# ε = ufl.sym(ufl.grad(model.momentum.vel))
# εe2 = ufl.inner(ε, ε) / 2

dx = msh.geometry.x[:,0] - msh0.geometry.x[:,0]
dy = msh.geometry.x[:,1] - msh0.geometry.x[:,1]

k = 100
msh.geometry.x[:,0] = msh0.geometry.x[:,0] + k*dx
msh.geometry.x[:,1] = msh0.geometry.x[:,1] + k*dy



tess = kr.plotting.get_triangulation(msh)

d2a = lambda f: kr.plotting.dolfinx_to_array(msh, f)
# ax.plot(*kr.plotting.get_outline(msh), lw=1.0,color='black',label='Iceberg outline')
# ax.plot(*kr.plotting.get_outline(msh0), lw=0.5, ls='--' , color='gray',label='Initial outline')

# ax.set_aspect('equal')



fig, ax = plt.subplots(figsize=(4,4))


A = mf.rate_factor(model.params.T)/model.params.A0

τe = 2*model.momentum.ε_eD
τe2 = 0.5*ufl.inner(τe,τe)
η = 1/(A*τe2)


τv = 2/A*mf.ε(model.momentum.vel)



ax.scatter(d2a(τv[0,1]), d2a(τe[0,1]), label = '0,1')
ax.scatter(d2a(τv[0,0]), d2a(τe[0,0]), label = '0,0')
ax.scatter(d2a(τv[1,1]), d2a(τe[1,1]), label = '1,1')
# ax.scatter(d2a(σvD[2,2]), d2a(σeD[2,2]), label = '2,2')
# ax.scatter(d2a(divsigmav[0]), d2a(divsigmae[0]), label = 'div 0')


#add line y=x
x = np.linspace(np.min(d2a(τv[0,1])), np.max(d2a(τv[0,1])), 100)
# ax.plot(x, x, 'k--', label='y=x')

# ax.set_aspect('equal')
# ax.set_xlim(-0.04,0.04)
ax.grid()
ax.legend()
#%%

σv = 2*mf.dev3(dεv) - model.momentum.p*ufl.Identity(3)
σe = es.cauchy_stress(model.momentum.ε_e,model.params.ν)

ν = model.params.ν
ε_e = mf.tensor_2d_to_3d(model.momentum.ε_e)
# σe = es.λoverμ(ν)*ufl.tr(ε_e)*ufl.Identity(3) + 2*ε_e

# σe = es.Koverμ(ν)*ufl.tr(ε_e)*ufl.Identity(3) + 2*ufl.dev(ε_e)
# σv = ufl.dev(σv)
# σe = ufl.dev(σe)


fig,ax = plt.subplots(figsize=(4,4))
ax.scatter(d2a(σv[0,1]), d2a(σe[0,1]), label = '0,1')
ax.scatter(d2a(σv[0,0]), d2a(σe[0,0]), label = '0,0')
ax.scatter(d2a(σv[1,1]), d2a(σe[1,1]), label = '1,1')
ax.scatter(d2a(σv[2,2]), d2a(σe[2,2]), label = '2,2')
# ax.scatter(d2a(-model.momentum.p), d2a(es.λoverμ(model.params.ν)*ufl.tr(model.momentum.ε_e)), label = 'p')
# ax.scatter(d2a(-3*model.momentum.p),d2a(3*es.Koverμ(model.params.ν)*ufl.tr(ε_e)), label = 'tr')
# ax.scatter(d2a
ax.grid()
ax.legend()
x = np.linspace(np.min(d2a(σv[2,2])), np.max(d2a(σv[2,2])), 100)
ax.plot(x, x, 'k--', label='y=x')
#%%
εe = ufl.dev(model.momentum.ε_e)
fig,ax = plt.subplots(figsize=(4,4))
# get means and stds of each component
means = np.zeros((3,3))
stds = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        means[i,j] = np.mean(d2a(εe[i,j]))
        stds[i,j] = np.std(d2a(εe[i,j]))
        ax.errorbar(means[i,j], means[i,j], xerr=stds[i,j], yerr=stds[i,j], fmt='o', label=f'{i},{j}')
        
mean_p = np.mean(d2a(ufl.tr(εe)))
std_p = np.std(d2a(ufl.tr(εe)))
# ax.errorbar(mean_p/3, mean_p/3, xerr=std_p/3, yerr=std_p/3, fmt='o', label='tr/3')
ax.legend()
#%%

fig,ax = plt.subplots(figsize=(4,4))

divu = ufl.div(v3to2(model.momentum.du))
# divu = ufl.tr(dεv)
norm = abs(ε3d(model.momentum.du)[0,0]) + abs(ε3d(model.momentum.du)[1,1]) 

ax.scatter(d2a(norm), d2a(divu))

# ax.set_aspect('equal')

#%%

fig,ax = plt.subplots(figsize=(4,4))
ax.scatter(d2a(η), d2a(η_v_s))

ax.set_aspect('equal')
x = np.linspace(np.min(d2a(η)), np.max(d2a(η)), 100)
ax.plot(x, x, 'k--', label='y=x')
ax.grid()

fig,ax = plt.subplots(figsize=(4,4))




σvDlin = 2*ε(model.momentum.vel)


σeD = 2*ufl.dev(mf.tensor_2d_to_3d(model.momentum.ε_e))
σeD = ufl.dev(es.cauchy_stress(mf.tensor_2d_to_3d(model.momentum.ε_e),model.params.ν))
# σeD = dev3(σe)
# η_e = viscosity_stress(σeD,n)
# ax.scatter(d2a(η_e), d2a(η_s))
ax.scatter(d2a(σvDlin[1,1]), d2a(σeD[1,1]), label = 'D 1,1', color='orange')
ax.scatter(d2a(σvDlin[0,0]), d2a(σeD[0,0]), label = 'D 0,0', color='blue')
ax.scatter(d2a(σvDlin[0,1]), d2a(σeD[0,1]), label = 'D 0,1', color='green')
ax.set_aspect('equal')
ax.grid()# %%




fig,ax = plt.subplots(figsize=(4,4))
σe2d = es.cauchy_stress(model.momentum.ε_e,model.params.ν)
# τe2d = mf.dev3(σe2d)
τe2d = 2*mf.dev3(model.momentum.ε_e)
# σzz = es.λoverμ(model.params.ν)*ufl.tr(model.momentum.ε_e)
ax.scatter(d2a(ufl.dev(σe)[1,1]), d2a(τe2d[1,1]), label = 'D 0,1', color='green')
ax.set_aspect('equal')
ax.grid()
# get gradient of line



# %%
