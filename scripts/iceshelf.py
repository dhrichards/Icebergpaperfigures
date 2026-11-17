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
import graded_mesh
from petsc4py import PETSc

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks above non dimensional")
parser.add_argument("--l", type=float, default=5, help="Regularization length scale")
parser.add_argument("--dtstar", type=float, default=1.0, help="Non-dimensional time step")
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=500, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--nt", type=int, default=50000, help="number of timesteps")
parser.add_argument("--T", type=float, default=-10, help="Temperature in Celsius at top")
parser.add_argument("--nondim_length", type=float, default=5, help="Length of iceberg")
parser.add_argument("--tol", type=float, default=5e-5, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=100, help="Maximum number of solver iterations")
parser.add_argument("--Kic", type=float, default=100, help="Kic")
parser.add_argument("--strength", type=float, default=100, help="Tensile strength at 0C")
parser.add_argument("--n", type=float, default=3.0, help="Glens law exponent")
parser.add_argument("--save_bp", action="store_true")


args = parser.parse_args()




path = './outputs'
os.makedirs(path, exist_ok=True)

lstar = args.l/args.height
small_size = lstar/args.cellfactor

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
model.max_its = 300

x = ufl.SpatialCoordinate(msh)
z = x[msh.geometry.dim-1]
model.params.T.value = args.T
model.params.A0.value = mf.rate_factor_np(args.T)*(0.5*0.1*900*9.8*500)**(3-args.n)
model.params.n.value = args.n
model.params.H.value = args.height
model.params.l.value = args.l
model.params.Kic.value = args.Kic*1e3
model.params.patm.value = 0.0
model.params.crack_level_above_sea.value = args.level*args.height
model.params.ρc = 0.1*900

model.params.σt.value = args.strength*1e3
model.params.dt.value = args.dtstar*model.params.τ_float


##Overload Kic for AT1
#Higher order
p = model.params
p.Kic = ufl.sqrt(2.0 * p.l * p.σt**2 / (1 - p.ν**2))
Kic_value = np.sqrt(2.0 * args.l * (args.strength*1e3)**2 / (1 - p.ν.value**2))/1e3

#Lower order
# p.Kic = ufl.sqrt(
#     (8.0/3.0) * (1-p.ge_tol) * p.l * p.σt**2 / (1-p.ν**2)
# )
# Kic_value = args.Kic


filename = (
    f"iceshelf_L{args.nondim_length:g}"
    f"_H{args.height:g}"
    f"_l{args.l:g}"
    f"_dtstar{args.dtstar:g}"
    f"_sigmac{args.strength:g}"
    f"_n{args.n:g}"
    f"_level{args.level:g}"
    f"_Kic{Kic_value:g}"
    f"_cellfactor{args.cellfactor:g}"
    f"_T{abs(args.T):g}"
    f"_{args.suffix}_"
)


if MPI.COMM_WORLD.rank == 0:
    print("ucstar: ", model.params.ucstar_float )
    print("τ_float: ", model.params.τ_float/(24*60*60) )
    print(path + "/" + filename)



def left_boundary(x):
    return np.isclose(x[0], 0)



u_bc = lambda V: [
                            bc.get_zero_bc(V.sub(0), left_boundary),
]

d_bc = lambda V: [
                bc.internal_bc(V, lambda x: ~refined_area(x), 0.0),
                ]


model.setup(kr.momentum.tensorquadrature.Tensor,
                           kr.damage.higherorder.AT1Bounded, [u_bc, d_bc])

model.momentum.initial_infinite_shelf()



t = 0.0

comm = model.msh.comm


s = ufl.dev(model.momentum.σ)
s_e2 = ufl.inner(s, s) / 2
σ0 = es.cauchy_stress(model.momentum.ε_e, model.params.ν)

output_fields = {
"du": model.momentum.du,
"d": model.damage.d,
"dprev2": model.damage.d_prev_it2,
"dprev": model.damage.d_prev_it,
"dprev3": model.damage.d_prev_it3,
"psi_plus": model.momentum.ψplus / model.params.ψcritstar,
"eps_e": model.momentum.ε_e,
"s": s,
"s_e2": s_e2,
"sig0": σ0,
"sig": model.momentum.σ,
"fluidity": model.momentum.fluidity_prev_it,
"fluidityused": model.momentum.fluidity_used,
"dq": model.momentum.dq,
    }


for i in range(1, args.nt):

    if comm.rank == 0:
        print("Iteration:", i, "time:", t / (24*60*60), "days")

    if i == 2:
        model.max_its = args.max_its

    
    flag, nits = model.solve_timestep(save=False, stop_bottom=True)

    failed = comm.allreduce(flag in (-1, 2), op=MPI.LOR)
    bottom_reached = comm.allreduce(flag in (3, 4), op=MPI.LOR)

    save_this_step = (
        failed or bottom_reached
        or i == 1 or i % 300 == 0 or nits > 30
    )
                
    if args.save_bp and save_this_step:
        model.write_checkpoint(
            path + "/" + filename + ".bp",
            t,
        )
    
    if save_this_step:
        kr.plotting.write_xdmf(
            path + "/" + filename + "run" + str(i) + ".xdmf",
            model.msh,
            list(output_fields.values()),
            list(output_fields.keys()),
            t=i,
            quadrature_degree=model.quad_degree,
        )

    if failed or bottom_reached:
        # Diagnostic output: do not rerun the constitutive update.
        kr.plotting.write_xdmf(
            path + "/" + filename + "failed" + str(i) + ".xdmf",
            model.msh,
            list(output_fields.values()),
            list(output_fields.keys()),
            t=i,
            quadrature_degree=model.quad_degree,
        )

        model.write_checkpoint(
                    path + "/" + filename + "end.bp",
                    t, append=False
                )

        if comm.rank == 0:
            print(
                f"Stopping at iteration {i}: "
                f'fixed_point returned flag {flag}',
                flush=True,
            )
    
        break


    t += float(model.params.dt.value)
    model.timestep()


    model.params.dt.value = min(
        1.1 * float(model.params.dt.value),
        args.dtstar * model.params.τ_float,
        )


if MPI.COMM_WORLD.rank == 0:
    print("time it:",  i)
    print("time t:",  t)
    print(path + "/" + filename)


   
