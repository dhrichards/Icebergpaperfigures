#%%
from mpi4py import MPI
import numpy as np
import ufl
import os
import dolfinx
from dolfinx import io, mesh, fem
import kraken.parameters as kp
import kraken.boundaryconditions as bc
import kraken.numerics.maths_functions as mf
import kraken.numerics.energy_splits as es
import kraken as kr
import argparse
from petsc4py import PETSc

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks above sea level (m)")
parser.add_argument("--lstar", type=float, default=0.005, help="Regularization length scale in meters")
parser.add_argument("--dtstar", type=float, default=1.0, help="Non-dimensional time step in days")
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=500, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--nt", type=int, default=10000, help="number of timesteps")
parser.add_argument("--T", type=float, default=-10, help="Temperature in Celsius at top")
parser.add_argument("--nondim_length", type=float, default=5, help="Length of iceberg")
parser.add_argument("--tol", type=float, default=5e-5, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=800, help="Maximum number of solver iterations")
parser.add_argument("--Kic", type=float, default=100, help="Kic")
parser.add_argument("--strength", type=float, default=200, help="Tensile strength at 0C")
parser.add_argument("--n", type=float, default=3.0, help="Glens law exponent")
parser.add_argument("--save_bp", type=bool, default=False, help="Save bp files")
parser.add_argument("--lfactor", type=float, default=2.0, help="Multiply l by in lower part of domain")
parser.add_argument("--mesh_smoothing", type=int, default=0, help="Mesh smoothing between timesteps")

args = parser.parse_args()


filename = "iceshelf_L" + str(args.nondim_length) + "_H" + str(args.height) \
                        + "_l" + str(args.lstar) \
                        + "_dtstar" + str(args.dtstar) \
                        + "_sigmac" + str(args.strength) \
                        + "_n" + str(args.n) \
                        + "_level" + str(args.level) \
                        + "_Kic" + str(args.Kic)\
                        + "_cellfactor" + str(args.cellfactor)\
                        + "_T" + str(abs(args.T)) \
                        + "_lfactor" + str(args.lfactor) \
                        + "_meshsmoothing" + str(args.mesh_smoothing) \
                        + "_" + args.suffix + "_"


path = './outputs'
os.makedirs(path, exist_ok=True)


small_size = args.lstar/args.cellfactor

large_size_aim = 0.2
n_div = int(np.log2(large_size_aim/small_size))
large_size = small_size*2**n_div

nx = int(args.nondim_length/large_size)
nz = int(1/large_size)
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                        [[0.0, 0.0],
                        [args.nondim_length, 1.0]],
                        [nx, nz],
                        cell_type=mesh.CellType.triangle)

def refined_area(x):
    return (x[0] > (args.nondim_length - 0.3))\
                |((x[1]>0.85)*(x[0]>(args.nondim_length-3)))

msh = kr.meshes.refine_by_area(msh,refined_area,n_div)


model = kr.base.Simulation(msh)
model.basal_friction = False
model.order = 2
model.quad_degree = 6

model.tol = args.tol
model.min_its = args.min_its
model.max_its = args.max_its

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T.value = args.T
model.params.A0.value = mf.rate_factor_np(args.T)*(0.5*0.1*900*9.8*500)**(3-args.n)
model.params.n.value = args.n
model.params.H.value = args.height
# model.params.l.value = args.lstar*args.height
model.params.Kic.value = args.Kic*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = args.level*args.height
model.params.ρc = dolfinx.fem.Constant(model.msh,0.1*900)
model.params.viscosity_tol.value = 1e-4

model.params.σt = args.strength*1e3
model.params.dt.value = args.dtstar*model.params.τ_float

def smoothstep(x, x_c, width):
    return 0.5*(1 + ufl.tanh((x-x_c)/width))

def smoothtransition(a, b, x, x_c, width):
    return a + (b-a)*smoothstep(x, x_c, width)

if args.lfactor>1.0:
    model.params.l = smoothtransition(args.lstar*args.height*args.lfactor, args.lstar*args.height, x[1], 1 - 0.125, 0.05)
else:
    model.params.l.value = args.lstar*args.height

if MPI.COMM_WORLD.rank == 0:
    print("ucstar: ", model.params.ucstar_float )
    print("τ_float: ", model.params.τ_float/(24*60*60) )
    print(path + "/" + filename)


# model.params.crack_level_above_sea = smoothtransition(
#     0,args.level, x[1], 1 - 0.25, 0.05
# )

def left_boundary(x):
    return np.isclose(x[0], 0)







u_bc = lambda V: [
                            bc.get_zero_bc(V.sub(0), left_boundary),
]

d_bc = lambda V: [
                bc.internal_bc(V, lambda x: ~refined_area(x), 0.0),
                ]

model.setup(kr.momentum.tensorquadrature.Tensor,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])

model.momentum.initial_infinite_shelf()


model.momentum.mesh_smoothing = bool(args.mesh_smoothing)

def crack(x,x_c,height=0.06):
    width = args.lstar/args.cellfactor*1
    return (x[0]>(x_c-width))*(x[0]<(x_c+width))*(x[1]>(1-height))

# end_crack_x_cs = np.linspace(args.nondim_length-2, args.nondim_length-0.15, 20)
end_crack_x_cs = args.nondim_length - np.arange(0.175,2,0.1)

height = 0.08
def end_cracks(x):
    val = np.zeros(x.shape[1],dtype=bool)
    for x_c in end_crack_x_cs:
        val += crack(x,x_c,height)
    return val

frozen_dofs = fem.locate_dofs_geometrical(
    (model.damage.W.sub(0), model.damage.D),
    lambda x: x[1]>0.95,
)


t = 0.0
model.momentum.solve()
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t)

model.momentum.creep_newton_weight.value = 1.0
model.params.dt.value = 100*model.params.τ_float
model.damage_on = False
for i in range(100):
    if MPI.COMM_WORLD.rank == 0:
            print("Relaxation it : ", i)
    model.momentum.solve()
    model.momentum.timestep()

# # model.damage.w.sub(0).interpolate(end_cracks)

model.damage_on = True
model.params.dt.value = args.dtstar*model.params.τ_float
model.momentum.creep_newton_weight.value = 0

newtonfactor=0.0
comm = model.msh.comm


easy_steps = 0
easy_nits = 3
easy_steps_required = 5
growth_factor = 1.25

for i in range(1, args.nt):

    if comm.rank == 0:
        print("Iteration:", i, "time:", t / (24*60*60), "days")

    failure_message = None
    nits = 0

    model.save_restart_state()
    trial_dt = float(model.params.dt.value)
    max_retries = 5

    for retry in range(max_retries + 1):
        save = retry > 0

        try:
            flag, nits = model.fixed_point(
                save=save, stop_bottom=True
            )
            break

        except (RuntimeError, PETSc.Error) as exc:
            easy_steps = 0
            PETSc.Sys.Print(f"Trial {retry} failed: {exc}")

            model.restore_restart_state()

            if retry == max_retries:
                raise

            trial_dt *= 0.5
            model.params.dt.value = trial_dt

            model.momentum.solver.destroy()
            model.momentum.setup_solver()

            PETSc.Sys.Print(
                f"Retrying viscoelastically: retry {retry + 1}, "
                f"dt={trial_dt:.6g}"
            )

    if i == 10:
        d_frozen = fem.Function(model.damage.D, name="frozen_damage")
        d_frozen.interpolate(model.damage.w.sub(0))
        d_frozen.x.scatter_forward()

        bc_d = lambda V: [fem.dirichletbc(d_frozen,frozen_dofs,V),
                          bc.internal_bc(V, lambda x: ~refined_area(x), 0.0)]
        model.damage.update_bcs(bc_d)
        

    # Ensure every rank takes the same output/exit branch.
    failed = comm.allreduce(flag != 1, op=MPI.LOR)

    

    s = ufl.dev(model.momentum.σ)
    s_e2 = ufl.inner(s, s) / 2
    σ0 = es.cauchy_stress(model.momentum.ε_e, model.params.ν)

    if failed:
        messages = comm.allgather(failure_message)

        if comm.rank == 0:
            message = next((m for m in messages if m), None)
            print(
                f"Stopping at iteration {i}: "
                f"{message or f'fixed_point returned flag {flag}'}",
                flush=True,
            )

        model.momentum.w.x.scatter_forward()
        model.damage.w.x.scatter_forward()

        # Diagnostic output: do not rerun the constitutive update.
        kr.plotting.write_xdmf(
            path + "/" + filename + "failed" + str(i) + ".xdmf",
            model.msh,
            [
                model.momentum.du,
                model.damage.d,
                model.damage.d_prev_it2,
                model.damage.d_prev_it,
                model.damage.d_prev_it3,
                model.momentum.ψplus / model.params.ψcritstar,
                model.momentum.ε_e,
                s,
                s_e2,
                σ0,
                model.momentum.σ,
                model.momentum.fluidity_prev_it,
                model.momentum.fluidity_used,
                model.momentum.dq,

            ],
            [
                "du", "d", "dprev2", "dprev", "dprev3",
                "psi_plus", "eps_e", "s", "s_e2", "sig0", "sig",
                "fluidity","fluidityused","dq"
            ],
            t=i,
            quadrature_degree=model.quad_degree,
        )

        break

    save_this_step = i == 1 or i % 10 == 0 or nits > 30

    if args.save_bp and save_this_step:
        model.write_checkpoint(
            path + "/" + filename + ".bp",
            t,
        )

    

    if save_this_step:
        kr.plotting.write_xdmf(
            path + "/" + filename + "run" + str(i) + ".xdmf",
            model.msh,
            [
                            model.momentum.du,
                            model.damage.d,
                            model.damage.d_prev_it2,
                            model.damage.d_prev_it,
                            model.damage.d_prev_it3,
                            model.momentum.ψplus / model.params.ψcritstar,
                            model.momentum.ε_e,
                            s,
                            s_e2,
                            σ0,
                            model.momentum.σ,
                            model.momentum.fluidity_prev_it,
                            model.momentum.fluidity_used,
                            model.momentum.dq,
            
                        ],
                        [
                            "du", "d", "dprev2", "dprev", "dprev3",
                            "psi_plus", "eps_e", "s", "s_e2", "sig0", "sig",
                            "fluidity","fluidityused","dq"
                        ],
            t=i,
            quadrature_degree=model.quad_degree,
        )


    t += float(model.params.dt.value)
    model.momentum.timestep()

    # # Count only easy steps that needed no retries.
    # if flag == 1 and retry == 0 and nits <= easy_nits:
    #     easy_steps += 1
    # else:
    #     easy_steps = 0

    # if easy_steps >= easy_steps_required:
    #     model.params.dt.value*=growth_factor
    #     easy_steps =0
    #     PETSc.Sys.Print(f"Increasing dt")


if MPI.COMM_WORLD.rank == 0:
    print("time it:",  i)
    print("time t:",  t)
    print(path + "/" + filename)


   
