#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
import dolfinx
from dolfinx import io, mesh
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lstar", type=float, default=0.02, help="Regularization length scale in meters")
parser.add_argument("--cellfactor", type=float, default=2, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=600, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--save_bp", action="store_true")
parser.add_argument("--nt", type=int, default=50, help="number of timesteps")
args = parser.parse_args()


filename = "ssa_H" + str(args.height) \
                        + "_l" + str(args.lstar) \
                    + "_cellfactor" + str(args.cellfactor)\
                            + "_" + args.suffix + "_"



def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], nondim_length)


path = './outputs'
os.makedirs(path, exist_ok=True)




nondim_length = 2
nondim_height = 1.0


# cell_size = 0.2 # large size
cell_size = args.lstar/args.cellfactor
nx = int((nondim_length)/cell_size)
if np.mod(nx,2) == 0:
    nx = nx + 1

nz = int(nondim_height/cell_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [nondim_length, nondim_height]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)


model = kr.base.Simulation(msh)

model.tol = 5e-6
model.min_its = 3
model.max_its = 1300

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T.value = -10.0
model.params.A0.value = mf.rate_factor_np(model.params.T.value)
model.params.H.value = args.height
model.params.l.value = args.lstar*args.height

model.params.Kic.value = 100*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = 0.0
model.params.σt.value = 200e3
model.params.ρc = 0.1*900

model.params.dt.value = 1.0*model.params.τ_float
if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)


dtstar = model.params.dt.value/model.params.τ_float
u_x = dtstar*nondim_length*(1/4)**3

u_bc = lambda V: [
                    bc.get_zero_bc(V.sub(0), left_boundary),
                    bc.get_bc(V.sub(0), right_boundary, u_x),
                    ]


d_bc = lambda V: [
                    bc.get_zero_bc(V, left_boundary),
                    bc.get_zero_bc(V, right_boundary),
                    ]


model.setup(kr.momentum.tensorquadrature.Tensor,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])

model.momentum.initial_infinite_shelf()


t = 0.0
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t, append=False)


model.damage_on = True


output_fields = {
"du": model.momentum.du,
"d": model.damage.d,
"dprev2": model.damage.d_prev_it2,
"dprev": model.damage.d_prev_it,
"dprev3": model.damage.d_prev_it3,
"psi_plus": model.momentum.ψplus / model.params.ψcritstar,
"eps_e": model.momentum.ε_e,
"sig": model.momentum.σ,
"fluidity": model.momentum.fluidity_prev_it,
"fluidityused": model.momentum.fluidity_used,
"dq": model.momentum.dq,
    }


for i in range(1, args.nt+1):
    t += model.params.dt.value
    flag,nits = model.fixed_point(save=True)


    if args.save_bp:
        model.write_checkpoint(path + "/" + filename +".bp", t)

    kr.plotting.write_xdmf(
                path + "/" + filename + "run" + str(i) + ".xdmf",
                model.msh,
                list(output_fields.values()),
                list(output_fields.keys()),
                t=i,
                quadrature_degree=model.quad_degree,
            )
    

    model.timestep()
        
 
if MPI.COMM_WORLD.rank == 0:
    print(path + "/" + filename)


   
