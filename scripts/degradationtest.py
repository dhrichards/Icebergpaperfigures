#%%
import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import kraken as kr
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
from kraken.momentum.tensorquadrature import Tensor


parser = argparse.ArgumentParser()
parser.add_argument('--lstar', type=float, default=0.005, help='Dimensionless regularisation length')
parser.add_argument('--cellfactor', type=float, default=2, help='Fine cell size = lstar/cellfactor')
parser.add_argument('--height', type=float, default=400, help='Ice thickness in metres')
parser.add_argument('--nondim_length', type=float, default=5)
parser.add_argument('--T', type=float, default=-10, help='Temperature in Celsius')
parser.add_argument('--level', type=float, default=-0.9, help='Crack water level relative to sea level, divided by H')
parser.add_argument('--tol', type=float, default=5e-6)
parser.add_argument('--min_its', type=int, default=3)
parser.add_argument('--max_its', type=int, default=800)
parser.add_argument('--suffix', default='')
parser.add_argument('--save_bp', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--relax_days', type=float, default=30, help='Undamaged evolution before fracture')
parser.add_argument('--dtstar_relax', type=float, default=10.0, help='Non-dimensional relaxation timestep')
parser.add_argument('--dtstar_damage', type=float, default=0.25, help='Non-dimensional damage timestep')
args = parser.parse_args()
if args.dtstar_relax <= 0 or args.dtstar_damage <= 0 or args.relax_days < 0:
    parser.error('Require dtstar_relax > 0, dtstar_damage > 0, relax_days >= 0')

path = Path('./outputs')
path.mkdir(exist_ok=True)

# Same mesh construction as iceshelf.py, refining only the upper strip.
small_size = args.lstar / args.cellfactor
large_size_aim = 0.2
n_div = int(np.log2(large_size_aim / small_size))
large_size = small_size * 2**n_div
nx = int(args.nondim_length / large_size)
nz = int(1 / large_size)
msh = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0., 0.], [args.nondim_length, 1.]],
    [nx, nz], cell_type=mesh.CellType.triangle,
)


def refined_area(x):
    return (x[0] > args.nondim_length - 2.5) & (x[1] > 0.75)


msh = kr.meshes.refine_by_area(msh, refined_area, n_div)


def left_boundary(x):
    return np.isclose(x[0], 0.)


u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
d_bc = lambda V: [bc.internal_bc(V, lambda x: ~refined_area(x), 0.)]

filename = path / (
    f'degradationtest_L{args.nondim_length:g}_H{args.height:g}'
    f'_l{args.lstar:g}_dtstar_relax{args.dtstar_relax:g}'
    f'_dtstar_damage{args.dtstar_damage:g}_cellfactor{args.cellfactor:g}'
    f'_T{args.T:g}_level{args.level:g}_relax{args.relax_days:g}d_{args.suffix}'
)
model = kr.base.Simulation(msh)
model.basal_friction = False
model.order = 2
model.quad_degree = 6
model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

model.params.T.value = args.T
model.params.A0.value = mf.rate_factor_np(args.T)
model.params.H.value = args.height
model.params.ρc = 0.1 * 900
model.params.Kic.value = 100e3
model.params.σt.value = 200e3
model.params.patm.value = 0.
model.params.crack_level_above_sea.value = args.level * args.height

model.params.l.value = args.lstar * args.height

model.params.viscosity_tol.value = 1e-4
model.setup(Tensor, kr.damage.higherorder.AT2, [u_bc, d_bc])
model.momentum.initial_infinite_shelf()
model.damage_on = False
model.momentum.creep_newton_weight.value = 1.0

DAY = 24 * 60 * 60
relax_until = args.relax_days * DAY
relax_dt = args.dtstar_relax * model.params.τ_float
damage_dt = args.dtstar_damage * model.params.τ_float
model.params.dt.value = relax_dt
t = 0.0
step = 0


output_fields = {
    'du': model.momentum.du,
    'd': model.damage.d,
    'psi_plus': model.momentum.ψplus / model.params.ψcritstar,
    'eps_e': model.momentum.ε_e,
    'sig': model.momentum.σ,
    'sigma_xx_Pa': model.params.pwc * model.momentum.σ[0, 0],
    'sigma_zz_Pa': model.params.pwc * model.momentum.σ[2, 2],
}

# Undamaged viscoelastic evolution for 30 days by default.
while t < relax_until - 1e-6:
    model.params.dt.value = min(relax_dt, relax_until - t)
    model.momentum.solve()
    t += float(model.params.dt.value)
    step += 1

    if MPI.COMM_WORLD.rank == 0:
        print('Relaxation step:', step, 'time:', t / DAY, 'days')

    # kr.plotting.write_xdmf(
    #     str(filename) + f'_relax_{step}.xdmf', model.msh,
    #     list(output_fields.values()), list(output_fields.keys()), t=t,
    #     quadrature_degree=model.quad_degree,
    # )
    model.timestep()


model.damage_on = True
model.momentum.creep_newton_weight.value = 0.0
model.params.dt.value = damage_dt

if MPI.COMM_WORLD.rank == 0:
    print('Damage step, time:', t / DAY, 'days')

flag, nits = model.fixed_point(save=False, stop_bottom=False)
failed = model.msh.comm.allreduce(flag not in (1, 4), op=MPI.LOR)
output_time = t if failed else t + float(model.params.dt.value)
label = 'failed' if failed else 'damage'

kr.plotting.write_xdmf(
    str(filename) + f'_{label}.xdmf', model.msh,
    list(output_fields.values()), list(output_fields.keys()), t=output_time,
    quadrature_degree=model.quad_degree,
)

if failed:
    raise RuntimeError(f'Damage step did not converge: flag={flag}, iterations={nits}')

# Save the resolved damage and displacement increment on the relaxed mesh.
# Do not call timestep(): retain the increment without moving the mesh.
if args.save_bp:
    model.write_checkpoint(str(filename) + '.bp', output_time, append=False)