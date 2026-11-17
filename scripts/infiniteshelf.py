#%%
import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import ufl
import kraken as kr
import kraken.boundaryconditions as bc


parser = argparse.ArgumentParser()
parser.add_argument("--l", type=float, default=5.0,
                    help="Regularisation length in metres")
parser.add_argument("--cellfactor", type=float, default=2.0,
                    help="Mesh size h = ell/cellfactor")
parser.add_argument("--height", type=float, default=600.0,
                    help="Ice thickness in metres")
parser.add_argument("--strength", type=float, default=100.0,
                    help="Tensile strength in KPa")
parser.add_argument("--suffix", default="")
parser.add_argument("--save_bp", action="store_true")
args = parser.parse_args()

path = Path("./outputs")
path.mkdir(exist_ok=True)

filename = path / (
    f"ssa_H{args.height:g}_l{args.l:g}"
    f"_cellfactor{args.cellfactor:g}_{args.suffix}_"
)

lstar = args.l / args.height
nondim_length = 2.0
cell_size = lstar / args.cellfactor

nx = int(nondim_length / cell_size)
if nx % 2 == 0:
    nx += 1
nz = int(1.0 / cell_size)

msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [nondim_length, 1.0]],
    [nx, nz],
    cell_type=mesh.CellType.triangle,
)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], nondim_length)


u_bc = lambda V: [
    bc.get_zero_bc(V.sub(0), left_boundary),
    bc.get_zero_bc(V.sub(0), right_boundary),
]

d_bc = lambda V: [
    bc.get_zero_bc(V, left_boundary),
    bc.get_zero_bc(V, right_boundary),
]

model = kr.base.Simulation(msh)
model.tol = 5e-5
model.min_its = 3
model.max_its = 1300
model.order = 1
model.quad_degree = 4

model.params.H.value = args.height
model.params.l.value = args.l
model.params.σt.value = args.strength*1e3
model.params.ρc = 0.1 * 900
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = 0.0

p = model.params
p.Kic = ufl.sqrt(2.0 * p.l * p.σt**2 / (1 - p.ν**2))

model.setup(
    kr.momentum.elasticlinearised.Elasticity,
    kr.damage.higherorder.AT1Bounded,
    [u_bc, d_bc],
)

model.momentum.initial_infinite_shelf()

if MPI.COMM_WORLD.rank == 0:
    print(filename)

flag, nits = model.fixed_point()

if args.save_bp:
    model.write_checkpoint(str(filename) + ".bp", 0.0, append=False)

output_fields = {
    "du": model.momentum.du,
    "d": model.damage.d,
    "psi_plus": model.momentum.ψplus / model.params.ψcritstar,
    "eps_e": model.momentum.ε_e,
}

kr.plotting.write_xdmf(
    str(filename) + ".xdmf",
    model.msh,
    list(output_fields.values()),
    list(output_fields.keys()),
    t=0.0,
    quadrature_degree=model.quad_degree,
)