"""Intact moving-mesh Kraken comparison with Christmann et al. (2019).

Reference: https://doi.org/10.1017/jog.2018.107, Sections 4--6.
This selects their CONSTANT-VISCOSITY reference case (Figs. 5--6),
not the additional n=3 experiment in Fig. 7. It retains Kraken's
constitutive formulation; it does not implement their finite-strain model.

Run, in the environment containing your tensorquadrature Kraken version:
    mpiexec -n 8 python christmann_2019.py
    mpiexec -n 8 python christmann_2019.py --years 1 --dt-days 0.5

Numerical choices specific to this driver (not prescribed by the paper):
one-second first step (or less if --dt-days is smaller), followed by a
constant --dt-days timestep. Stop at the first step reaching --years;
the final time may exceed it by less than one timestep. COMSOL used
automatic timesteps.
Use smaller timesteps/mesh sizes to check convergence before comparison.

Detailed output is run.bp: Kraken checkpoints, including the initial mesh
at t=0 for reference coordinates (before the elastic loading step).
Save only the initial state, first elastic step, and comparison years listed
in save_years. Each target uses the first fixed step reaching that time.
BP timestamps are seconds; snapshots.csv also lists days and years.
Mesh coordinates and elastic strain retain Kraken's nondimensional scaling;
multiply them by H and ucstar, respectively. Reconstruct stress using
es.cauchy_stress(eps_e_prev_time, nu) * pc.
Optional --debug-xdmf writes only sigma_xx in Pa for quick inspection;
its cell-averaged values are not the detailed plotting data.
Requires Simulation.dx to be the quadrature-aware property discussed above.
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI
import numpy as np
from dolfinx import mesh

import kraken as kr
import kraken.boundaryconditions as bc
import kraken.numerics.energy_splits as es


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--years", type=float, default=30.0)
parser.add_argument("--dt-days", type=float, default=1.0,
                    help="Constant timestep in days after the short first step.")
parser.add_argument("--output", default="outputs/christmann_2019")
parser.add_argument("--debug-xdmf", action="store_true",
                    help="Also write a minimal XDMF stress field for inspection.")
args = parser.parse_args()

comm = MPI.COMM_WORLD
path = Path(args.output)
path.mkdir(parents=True, exist_ok=True)

DAY = 86400.0
YEAR = 365.25 * DAY  # Explicit time-unit convention for this driver.
L = 5000.0
H = 100.0
ρi = 910.0
ρw = 1028.0
g = 9.81
E = 9.0e9
ν = 0.325
η = 1.0e14
μ = E / (2.0 * (1.0 + ν))
t_M = η / μ
draft = H * ρi / ρw

# Mesh coordinates are (x/H, z/H), with the initial base at z/H=0.
# Sea level is at draft/H, rather than z/H=0 as in the paper.
# The coarse triangle diagonal is <=50 m. Five refinements give <=2 m
# throughout the last kilometre, including its upper surface and front.
# This is a conservative refinement region, not their exact COMSOL mesh.
coarse_max = 50.0
fine_max = 5.0
spacing = coarse_max / np.sqrt(2.0)
nx = int(np.ceil(L / spacing))
nz = int(np.ceil(H / spacing))
msh = mesh.create_rectangle(
    comm, [[0.0, 0.0], [L / H, 1.0]], [nx, nz],
    cell_type=mesh.CellType.triangle,
)


def refined_area(x):
    return x[0] >= (L - 1000.0) / H


n_refine = int(np.ceil(np.log2(coarse_max / fine_max)))
msh = kr.meshes.refine_by_area(msh, refined_area, n_refine)

model = kr.base.Simulation(msh)
model.basal_friction = False
model.damage_on = False
model.order = 2
model.quad_degree = 6
# Simulation.dx is a property: quadrature settings are read at setup.

p = model.params
p.H.value = H
p.ρi.value = ρi
p.ρw.value = ρw
p.g.value = g
p.E.value = E
p.ν.value = ν
p.patm.value = 0.0
p.sea_level = draft
p.ρc = ρi  # Numerical scaling only; no artificial physical density contrast.

# Kraken: eps_v_dot = A0 * tau_eff**(n-1) * dev(sigma).
# Therefore n=1, A0=1/(2*eta) gives the paper's constant viscosity.
p.n.value = 1.0
p.A0.value = 1.0 / (2.0 * η)
p.ge_tol.value = 0.0  # Exactly unit degradation at d=0.
p.viscosity_tol.value = 0.0
p.dt.value = 1.0


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def u_bc(V):
    # Normal displacement fixed upstream; vertical motion remains free.
    return [bc.get_zero_bc(V.sub(0), left_boundary)]


# Keep the standard damage object for API compatibility, but never solve it.
# Its zero initial field stays zero; strength, toughness and length scale
# have no role in this intact calculation.
model.setup(kr.momentum.tensorquadrature.Tensor,
            kr.damage.higherorder.AT2, [u_bc, lambda V: []])
model.momentum.mesh_smoothing = False

# No initial_infinite_shelf() and no extra uncommitted momentum.solve().
# The first short solve starts from zero stored elastic strain and includes
# the front loading. It approximates the paper's instantaneous elastic state.
model.momentum.ε_e_prev_time.x.array[:] = 0.0
model.momentum.ε_e_prev_time.x.scatter_forward()

bp_file = str(path / "run.bp")


def save_state(step, t):
    model.write_checkpoint(bp_file, t, append=(step != 0))

    if args.debug_xdmf:
        σ = es.cauchy_stress(model.momentum.ε_e_prev_time, p.ν) * p.pc
        kr.plotting.write_xdmf(
            str(path / f"debug_{step:06d}.xdmf"), msh,
            [σ[0, 0]], ["sigma_xx_Pa"],
            t=t / YEAR, quadrature_degree=model.quad_degree,
        )
    if comm.rank == 0:
        with (path / "snapshots.csv").open("a") as f:
            f.write(f"{step},{t:.16g},{t / DAY:.16g},{t / YEAR:.16g}\n")


dt = args.dt_days * DAY
first_dt = min(1.0, dt, args.years * YEAR)
nt = 1 + int(np.ceil((args.years * YEAR - first_dt) / dt))
save_years = [1, 2, 4, 10, 30]
save_steps = {1} | {
    1 + int(np.ceil((year * YEAR - first_dt) / dt))
    for year in save_years if year <= args.years
}

if comm.rank == 0:
    print(f"L={L:g} m, H={H:g} m, draft={draft:.6f} m")
    print(f"eta={η:.3e} Pa s, A0={float(p.A0.value):.3e} Pa^-1 s^-1")
    print(f"Maxwell time={t_M / DAY:.6f} days; first step <=1 s")
    with (path / "parameters.json").open("w") as f:
        json.dump({
            "L_m": L, "H_m": H, "rho_i": ρi, "rho_w": ρw,
            "g": g, "E_Pa": E, "nu": ν, "draft_m": draft,
            "n": float(p.n.value), "A0": float(p.A0.value),
            "ucstar": float(p.ucstar_float), "pc_Pa": ρi * g * H,
            "order": model.order, "quadrature_degree": model.quad_degree,
            "quadrature_scheme": model.quad_scheme,
            "bp_time_unit": "seconds", "seconds_per_year": YEAR,
        }, f, indent=2)
    with (path / "snapshots.csv").open("w") as f:
        f.write("step,time_seconds,time_days,time_years\n")

t = 0.0
save_state(0, t)  # Preserve the initial mesh for material coordinates.
for step in range(1, nt + 1):
    p.dt.value = first_dt if step == 1 else dt

    # n=1 has no lagged nonlinear fluidity: one converged momentum solve
    # suffices. No damage-based fixed-point stopping test is involved.
    model.momentum.solve()
    reason = model.momentum.solver.getConvergedReason()
    if comm.allreduce(reason <= 0, op=MPI.LOR):
        raise RuntimeError(f"Momentum solve failed at t={t / DAY:g} days; SNES={reason}")

    model.timestep()  # Commit elastic history and move the mesh exactly once.
    t = first_dt + (step - 1) * dt
    if step in save_steps:
        save_state(step, t)

    if comm.rank == 0:
        print(f"Step {step}: t={t / DAY:.6f} days, dt={float(p.dt.value) / DAY:.6g} days", flush=True)

if comm.rank == 0:
    print(f"Finished at {t / YEAR:g} years. Outputs: {path}")