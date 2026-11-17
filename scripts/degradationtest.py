# %%
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
parser.add_argument('--l', type=float, default=2.0,
                    help='Regularisation length in metres')
parser.add_argument('--cellfactor', type=float, default=2.0,
                    help='Fine cell size = l/cellfactor')
parser.add_argument('--height', type=float, default=400.0,
                    help='Ice thickness in metres')
parser.add_argument('--nondim_length', type=float, default=5.0)
parser.add_argument('--T', type=float, default=-10.0,
                    help='Temperature in Celsius')
parser.add_argument('--level', type=float, default=-0.9,
                    help='Crack water level relative to sea level, divided by H')
parser.add_argument('--tol', type=float, default=5e-5)
parser.add_argument('--min_its', type=int, default=3)
parser.add_argument('--max_its', type=int, default=800)
parser.add_argument('--suffix', default='')
parser.add_argument('--save_bp', action=argparse.BooleanOptionalAction,
                    default=True)
parser.add_argument('--relax_days', type=float, default=30.0,
                    help='Undamaged evolution before fracture')
parser.add_argument('--dtstar_relax', type=float, default=10.0,
                    help='Non-dimensional relaxation timestep')
parser.add_argument('--dtstar_damage', type=float, default=0.25,
                    help='Non-dimensional damage timestep')

# AT1 links strength, toughness and l: prescribe either strength or toughness.
material = parser.add_mutually_exclusive_group()
material.add_argument('--strength', type=float, default=None,
                      help='Tensile strength in kPa; default 200 if Kic is omitted')
material.add_argument('--Kic', type=float, default=None,
                      help='Fracture toughness in kPa sqrt(m); derives strength')

args = parser.parse_args()

if min(args.l, args.cellfactor, args.height, args.nondim_length) <= 0:
    parser.error('Require l, cellfactor, height and nondim_length > 0')
if args.dtstar_relax <= 0 or args.dtstar_damage <= 0 or args.relax_days < 0:
    parser.error('Require dtstar_relax > 0, dtstar_damage > 0, relax_days >= 0')
if args.strength is not None and args.strength <= 0:
    parser.error('Require strength > 0')
if args.Kic is not None and args.Kic <= 0:
    parser.error('Require Kic > 0')

path = Path('./outputs')
path.mkdir(exist_ok=True)

# The mesh is non-dimensional; l is supplied in metres.
lstar = args.l / args.height
small_size = lstar / args.cellfactor
large_size_aim = 0.2
n_div = max(0, int(np.floor(np.log2(large_size_aim / small_size))))
large_size = small_size * 2**n_div
nx = max(1, int(np.ceil(args.nondim_length / large_size)))
nz = max(1, int(np.ceil(1.0 / large_size)))

msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [args.nondim_length, 1.0]],
    [nx, nz],
    cell_type=mesh.CellType.triangle,
)


def refined_area(x):
    return (x[0] > args.nondim_length - 2.5) & (x[1] > 0.75)


if n_div:
    msh = kr.meshes.refine_by_area(msh, refined_area, n_div)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]

# Damage is permitted only in the refined upper strip.
d_bc = lambda V: [
    bc.internal_bc(V, lambda x: ~refined_area(x), 0.0),
]

model = kr.base.Simulation(msh)
model.basal_friction = False
model.order = 2
model.quad_degree = 6
model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

p = model.params
p.T.value = args.T
p.A0.value = mf.rate_factor_np(args.T)
p.H.value = args.height
p.l.value = args.l
p.ρc = 0.1 * 900
p.patm.value = 0.0
p.crack_level_above_sea.value = args.level * args.height
p.viscosity_tol.value = 1e-4

# Higher-order AT1 calibration, as in iceshelf.py.
nu = float(p.ν.value)
if args.Kic is None:
    strength_kPa = 200.0 if args.strength is None else args.strength
    Kic_kPa_sqrt_m = strength_kPa * np.sqrt(
        2.0 * args.l / (1.0 - nu**2)
    )
else:
    Kic_kPa_sqrt_m = args.Kic
    strength_kPa = Kic_kPa_sqrt_m * np.sqrt(
        (1.0 - nu**2) / (2.0 * args.l)
    )

p.σt.value = strength_kPa * 1e3
p.Kic.value = Kic_kPa_sqrt_m * 1e3

filename = path / (
    f'degradationtest_AT1_L{args.nondim_length:g}_H{args.height:g}'
    f'_l{args.l:g}_strength{strength_kPa:g}_Kic{Kic_kPa_sqrt_m:g}'
    f'_dtstar_relax{args.dtstar_relax:g}'
    f'_dtstar_damage{args.dtstar_damage:g}_cellfactor{args.cellfactor:g}'
    f'_T{args.T:g}_level{args.level:g}_relax{args.relax_days:g}d'
    f'_{args.suffix}'
)

DAY = 24 * 60 * 60
relax_until = args.relax_days * DAY
relax_dt = args.dtstar_relax * p.τ_float
damage_dt = args.dtstar_damage * p.τ_float
p.dt.value = relax_dt

model.setup(Tensor, kr.damage.higherorder.AT1Bounded, [u_bc, d_bc])
model.momentum.initial_infinite_shelf()
model.damage_on = False
model.momentum.creep_newton_weight.value = 1.0

if MPI.COMM_WORLD.rank == 0:
    print(filename)
    print(
        f'l = {args.l:g} m, lstar = {lstar:g}, '
        f'strength = {strength_kPa:g} kPa, '
        f'Kic = {Kic_kPa_sqrt_m:g} kPa sqrt(m)'
    )

t = 0.0
step = 0

output_fields = {
    'du': model.momentum.du,
    'd': model.damage.d,
    'psi_plus': model.momentum.ψplus / p.ψcritstar,
    'eps_e': model.momentum.ε_e,
    'sig': model.momentum.σ,
    'sigma_xx_Pa': p.pwc * model.momentum.σ[0, 0],
    'sigma_zz_Pa': p.pwc * model.momentum.σ[2, 2],
}

# Undamaged viscoelastic evolution.
while t < relax_until - 1e-6:
    p.dt.value = min(relax_dt, relax_until - t)
    model.momentum.solve()
    t += float(p.dt.value)
    step += 1

    if MPI.COMM_WORLD.rank == 0:
        print('Relaxation step:', step, 'time:', t / DAY, 'days')

    model.timestep()

# One coupled viscoelastic fracture step on the relaxed configuration.
model.damage_on = True
model.momentum.creep_newton_weight.value = 0.0
p.dt.value = damage_dt

if MPI.COMM_WORLD.rank == 0:
    print('Damage step, time:', t / DAY, 'days')

flag, nits = model.fixed_point(save=False, stop_bottom=False)
failed = model.msh.comm.allreduce(flag not in (1, 4), op=MPI.LOR)
output_time = t if failed else t + float(p.dt.value)
label = 'failed' if failed else 'damage'

kr.plotting.write_xdmf(
    str(filename) + f'_{label}.xdmf',
    model.msh,
    list(output_fields.values()),
    list(output_fields.keys()),
    t=output_time,
    quadrature_degree=model.quad_degree,
)

if failed:
    raise RuntimeError(
        f'Damage step did not converge: flag={flag}, iterations={nits}'
    )

# Retain the displacement increment on the relaxed mesh.
# Do not call timestep() before saving.
if args.save_bp:
    model.write_checkpoint(str(filename) + '.bp', output_time, append=False)