#%%
import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import kraken as kr
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
from kraken.momentum.elastic import (
    Elasticity, ElasticityDegradedBodyForce, ElasticityHybrid,
)

parser = argparse.ArgumentParser()
parser.add_argument('--case', choices=['all', 'full', 'degraded_body', 'hybrid'], default='all')
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
args = parser.parse_args()

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
    return x[1] > 0.75


msh = kr.meshes.refine_by_area(msh, refined_area, n_div)


def left_boundary(x):
    return np.isclose(x[0], 0.)


u_bc = lambda V: [bc.get_zero_bc(V.sub(0), left_boundary)]
d_bc = lambda V: [bc.internal_bc(V, lambda x: ~refined_area(x), 0.)]

solvers = {
    'full': Elasticity,
    'degraded_body': ElasticityDegradedBodyForce,
    'hybrid': ElasticityHybrid,
}
cases = solvers if args.case == 'all' else {args.case: solvers[args.case]}

# Each case starts independently on the same fixed mesh.
for case, solver in cases.items():
    filename = path / (
        f'degradationtest_L{args.nondim_length:g}_H{args.height:g}'
        f'_l{args.lstar:g}_cellfactor{args.cellfactor:g}'
        f'_T{args.T:g}_level{args.level:g}_{case}_{args.suffix}'
    )
    model = kr.base.Simulation(msh)
    model.basal_friction = False
    model.order = 1
    model.quad_degree = 4
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

    model.setup(solver, kr.damage.higherorder.AT2, [u_bc, d_bc])
    model.momentum.initial_infinite_shelf()
    model.momentum.solve()
    if args.save_bp:
        model.write_checkpoint(str(filename) + '.bp', 0., append=False)

    model.damage_on = True
    # Seek the converged static response, without stopping at basal damage.
    flag, nits = model.fixed_point(save=True, stop_bottom=False)
    if flag in (1, 4):
        model.momentum.timestep()
    if MPI.COMM_WORLD.rank == 0:
        print(f'{case}: flag={flag}, iterations={nits}, output={filename}')
        if flag not in (1, 4):
            print('WARNING: saved final iterate is not a converged elastic response.')

    # Records 0 and 1 identify loading stages, not elapsed physical time.
    if args.save_bp:
        model.write_checkpoint(str(filename) + '.bp', 1.)

    fields = {
        'du': model.momentum.du,
        'd': model.damage.d,
        'psi_plus': model.momentum.ψplus / model.params.ψcritstar,
        'eps_e': model.momentum.ε_e,
        'sig': model.momentum.σ,
        'u_m': model.params.uc * model.momentum.u,
        'sigma_xx_Pa': model.params.pwc * model.momentum.σ[0, 0],
        'sigma_zz_Pa': model.params.pwc * model.momentum.σ[2, 2],
    }
    kr.plotting.write_xdmf(
        str(filename) + '.xdmf', msh,
        list(fields.values()), list(fields), t=1.,
        quadrature_degree=model.quad_degree,
    )