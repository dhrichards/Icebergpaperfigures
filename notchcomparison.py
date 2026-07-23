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



path = '~/Documents/kraken3/outputs'

notchbp = path + '/notched.bp'
no_notchbp = path + '/no_notch.bp'

mshs = []
i=1

fig,ax = plt.subplots(1,2,figsize=(7,3))
for filename in [notchbp, no_notchbp]:

    times =adios4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, "w_momentum")
    msh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, time=times[i])


    model = kr.base.Simulation(msh, kr.momentum.totalelastic.SemiLagrangian,
                        kr.damage.higherorder.HigherOrder)

    model.read_checkpoint(filename, t=times[i])
