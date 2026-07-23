#%%
from dolfinx import mesh, fem
import ufl
import numpy as np
from mpi4py import MPI
import basix.ufl as bufl
import adios4dolfinx
from matplotlib import tri
from matplotlib import pyplot as plt


def scalar_to_array(msh, f):
    CG1 = fem.functionspace(msh, ("CG", 1))
    # Put onto CG1 space

    # if hasattr(f,"ufl_function_space") and f.ufl_element().degree == 1:
    #             return f.x.array
    # else:
    f_CG1 = fem.Function(CG1)
    f_CG1.interpolate(fem.Expression(f, CG1.element.interpolation_points()))
    return f_CG1.x.array[:]
    

def vector_to_array(msh, f):
    fis = []
    for i in range(f.ufl_shape[0]):
        fis.append(scalar_to_array(msh, f.sub(i)))

    return fis

def dolfinx_to_array(msh, f):
    if len(f.ufl_shape) == 0:
        return scalar_to_array(msh, f)
    elif len(f.ufl_shape) == 1:
        return vector_to_array(msh, f)
    else:
        raise NotImplementedError("Only scalar and vector functions are supported.")
            


msh = adios4dolfinx.read_mesh("../elastictest.bp",MPI.COMM_WORLD)

U = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
d_el = bufl.element("CG", msh.basix_cell(), 2)
d_el_mixed = bufl.mixed_element([d_el, d_el])

W = fem.functionspace(msh, d_el_mixed)
u = fem.Function(U, name="u")


adios4dolfinx.read_function("../elastictest.bp", u, name="u")

w = fem.Function(W, name="w")
adios4dolfinx.read_function("../elastictest.bp", w, name="w")


# CG1 = fem.functionspace(msh, ("CG", 1))

# ux = fem.Function(CG1)
# uz = fem.Function(CG1)
# d = fem.Function(CG1)

# ux.interpolate(fem.Expression(u.sub(0),
#                              CG1.element.interpolation_points()))
# uz.interpolate(fem.Expression(u.sub(1),
#                              CG1.element.interpolation_points()))

# d.interpolate(fem.Expression(w.sub(0),
#                              CG1.element.interpolation_points()))




#%%
fig,ax = plt.subplots(figsize=(6,6))

d = dolfinx_to_array(msh, w.sub(0))
# u = dolfinx_to_array(msh, u)

x = msh.geometry.x[:,0]
z = msh.geometry.x[:,1]

x_def = x + u[0]
z_def = z + u[1]

connty = msh.topology.connectivity(2, 0)
connty_array = np.array([connty.links(i) 
        for i in range(connty.num_nodes)])
tess = tri.Triangulation(
        x_def, 
        z_def, 
        triangles=connty_array)


# ax.set_xlim(1e-1, 27)
ax.set_ylim(-4.1,4.5)
#axis off

## plot outline
# ax.triplot(tess, color='black', lw=0.3)

edges = np.sort(np.vstack([
    tess.triangles[:, [0, 1]],
    tess.triangles[:, [1, 2]],
    tess.triangles[:, [2, 0]],
]), axis=1)

# Count occurrences of edges
unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
boundary_edges = unique_edges[counts == 1]

# Build an array of NaN-separated segments
X = np.full((3 * len(boundary_edges),), np.nan)
Y = np.full((3 * len(boundary_edges),), np.nan)

for i, e in enumerate(boundary_edges):
    X[3*i:3*i+2] = x_def[e]
    Y[3*i:3*i+2] = z_def[e]

# Plot all in one go with a single label
ax.plot(X, Y, lw=1.0, label='$l = ' + str(1) + '$ m')
