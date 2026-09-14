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
parser.add_argument("--level", type=float, default=0.00, help="Water level in cracks above sea level (m)")
parser.add_argument("--lstar", type=float, default=0.005, help="Regularization length scale in meters")
parser.add_argument("--dtstar", type=float, default=1.0, help="Non-dimensional time step in days")
parser.add_argument("--cellfactor", type=float, default=1, help="Mesh cell size factor")
parser.add_argument("--height", type=float, default=500, help="Height of iceberg in meters")
parser.add_argument("--suffix", type=str, default="", help="suffix for filename")
parser.add_argument("--nt", type=int, default=10000, help="number of timesteps")
parser.add_argument("--T", type=float, default=-10, help="Temperature in Celsius at top")
parser.add_argument("--nondim_length", type=float, default=5, help="Length of iceberg")
parser.add_argument("--tol", type=float, default=5e-6, help="Solver tolerance")
parser.add_argument("--min_its", type=int, default=1, help="Minimum number of solver iterations")
parser.add_argument("--max_its", type=int, default=800, help="Maximum number of solver iterations")
parser.add_argument("--Kic", type=float, default=100, help="Kic")
parser.add_argument("--strength", type=float, default=200, help="Tensile strength at 0C")
parser.add_argument("--n", type=float, default=3.0, help="Glens law exponent")
parser.add_argument("--save_bp", type=bool, default=False, help="Save bp files")
parser.add_argument("--lfactor", type=float, default=2.0, help="Multiply l by in lower part of domain")
parser.add_argument("--mesh_smoothing", type=int, default=0, help="Mesh smoothing between timesteps")
parser.add_argument("--seed_cracks", type=int, default=1, help="Seed cracks in domain")

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
                        + "_seedcracks" + str(args.seed_cracks) \
                        + "_" + args.suffix + "_"


path = './outputs'
os.makedirs(path, exist_ok=True)



msh = kr.meshes.fenicsx_refined_mesh(args.nondim_length, args.lstar/args.cellfactor, 0.5, large_size=0.2, top_fine_length=2.5, htop2 =1.1)
model = kr.base.Simulation(msh)
model.basal_friction = False

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
model.params.crack_level_above_sea.value = args.level
model.params.ρc = dolfinx.fem.Constant(model.msh,0.1*900)
model.params.viscosity_tol.value = 1e-5

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
                            bc.get_zero_bc(V.sub(0).sub(0), left_boundary),
                            bc.get_zero_bc(V.sub(1).sub(0), left_boundary),
                            ]


def fixed(x):
    return (x[0]<(args.nondim_length -0.5))*(x[1]<0.9) | (x[0]<(args.nondim_length - 2.0))




d_bc = lambda V: [bc.internal_bc(V, fixed, 0.0),
                #   bc.internal_bc(V, end_cracks, 1.0),
                #   bc.internal_bc(V, lambda x: (x[1]>(1-height))*(1-end_cracks(x)), 0.0),
                ]


model.setup(kr.momentum.mixed.SemiLagrangianEpsilon,
                           kr.damage.higherorder.AT2, [u_bc, d_bc])

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




t = 0.0
model.momentum.solve()
if args.save_bp:
    model.write_checkpoint(path + "/" + filename +".bp", t, append=False)



if args.seed_cracks:
    model.damage.w.sub(0).interpolate(end_cracks)

model.damage_on = True


for i in range(1,args.nt):

    if MPI.COMM_WORLD.rank == 0:
        print("Iteration: ", i, "time: ", t/(24*60*60), "days")

    flag,nits = model.fixed_point(save=True, stop_bottom=False)

    t += model.params.dt.value
    if args.save_bp:
        if i == 1 or i % 20 == 0 or flag == -1 or nits > 30:
            model.write_checkpoint(path + "/" + filename +".bp", t, append=True)


    η0 = mf.viscosity(ufl.dev(mf.ε(model.momentum.vel_prev_it)), 3.0, 1e-19)


    if i ==1 or i % 50 == 0 or flag == -1 or nits > 10:
        kr.plotting.write_xdmf(path + "/" + filename +"run" + str(i) + ".xdmf",
                                model.msh, [model.momentum.u,model.damage.d,model.damage.d_prev_it2,model.damage.d_prev_it,model.damage.d_prev_it3,
                                        model.momentum.u_v, model.momentum.u_e,
                                        model.momentum.ψplus/model.params.ψcritstar,
                                        model.momentum.ε_e,
                                        model.params.Gc,
                                        model.momentum.crack_pressure(model.momentum.du),
                                        η0,
                                        ],
                                        ["u","d","dprev2","dprev","dprev3",
                                        "uv","ue",
                                        "psi_plus",
                                        "eps_e",
                                        "Gc",
                                        "p_c",
                                        "eta",
                                        ],
                                    t=i)
        
    if flag == -1:
        break

    
    model.timestep()
    # model.momentum.timestep()


if MPI.COMM_WORLD.rank == 0:
    print("time it:",  i)
    print("time t:",  t/(24*3600))
    print(path + "/" + filename)

if args.save_bp == False:
    model.write_checkpoint(path + "/" + filename +"end.bp", t, append=False)
   
