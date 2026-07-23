#%%
from dolfinx import fem
import adios4dolfinx
from mpi4py import MPI
import numpy as np
from matplotlib import tri
import matplotlib.pyplot as plt
import basix.ufl as bufl

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
            

## latex
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif",
                     "font.serif": ["Computer Modern Roman"],
                     "font.size": 10})

plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"

ls = [8.0,4.0,2.0]#,1,0.5]
parts = ['nopsicrit','normal']
fig, axs = plt.subplots(2,1, figsize=(6,4))

for ax in axs:

    # draw recetangle
    rect = plt.Rectangle((0, -0.9),
                         80/3,
                         1,
                         linewidth=0.5,
                         edgecolor='black',
                         facecolor='none',
                         linestyle='--',
                        #  label='Undeformed'
                         )
    ax.add_patch(rect)


for ax, part in zip(axs, parts):
    ax.axis('off')
    ax.set_aspect(aspect=1)
    for l in ls:
        filename = part + 'elastic_l' + str(l) + '.bp'
        msh = adios4dolfinx.read_mesh(filename,MPI.COMM_WORLD)

        U = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
        d_el = bufl.element("CG", msh.basix_cell(), 2)
        d_el_mixed = bufl.mixed_element([d_el, d_el])
        W = fem.functionspace(msh, d_el_mixed)
        u_dolf = fem.Function(U, name="u")
        w = fem.Function(W, name="w")

        adios4dolfinx.read_function(filename, u_dolf, name="u")
        adios4dolfinx.read_function(filename, w, name="w")

        x,y = msh.geometry.x[:,0], msh.geometry.x[:,1]

        ux = scalar_to_array(msh, u_dolf.sub(0))
        uz = scalar_to_array(msh, u_dolf.sub(1))
        d = scalar_to_array(msh, w.sub(0))
        
    
        connty = msh.topology.connectivity(2, 0)
        connty_array = np.array([connty.links(i) 
                        for i in range(connty.num_nodes)])

        x = x + ux
        y = y + uz
        tess = tri.Triangulation(
                x, 
                y, 
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
            X[3*i:3*i+2] = x[e]
            Y[3*i:3*i+2] = y[e]

        # Plot all in one go with a single label
        ax.plot(X, Y, lw=1.0, label='$l = ' + str(l) + '$ m')

    #legend outside right
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.text(0.05, 0.8, '(' + ('a' if part=='' else 'b') + ')', transform=ax.transAxes, va='top', ha='left')

model2 = r"$-\nabla \cdot \bm{\hat{\sigma}}= g \bm{f} + p_w \nabla (g) $"
model1 = r"$-\nabla \cdot \bm{\hat{\sigma}}= \bm{f} + p_w \nabla_S (g) $"
axs[0].text(0.5, 0.8, model1, transform=axs[0].transAxes, va='top', ha='center')
axs[1].text(0.5, 0.8, model2, transform=axs[1].transAxes, va='top', ha='center')


plt.tight_layout()

plt.savefig('comparedisplacements_outline.pdf', bbox_inches='tight')

#%%
from matplotlib import gridspec
# fig,axs = plt.subplots(len(ls),len(parts), figsize=(8,3))
# use grid spec, one extra "2 column" for to show inset


fig = plt.figure(figsize=(8, 4))

# 4 rows total (3 for A–F, 1 for G), 2 columns
gs = gridspec.GridSpec(len(ls)+1, len(parts))


zoom_x = [24.5, 26.7]
zoom_y = [-0.2, 0.1]

labels = ['a','c','e','g','i','b','d','f','h','j']

for i in range(len(parts)):
    part = parts[i]
    for j in range(len(ls)):
        l = ls[j]
        filename = part + 'elastic_l' + str(l) + '.bp'
        msh = adios4dolfinx.read_mesh(filename,MPI.COMM_WORLD)

        U = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
        d_el = bufl.element("CG", msh.basix_cell(), 2)
        d_el_mixed = bufl.mixed_element([d_el, d_el])
        W = fem.functionspace(msh, d_el_mixed)
        u_dolf = fem.Function(U, name="u")
        w = fem.Function(W, name="w")

        adios4dolfinx.read_function(filename, u_dolf, name="u")
        adios4dolfinx.read_function(filename, w, name="w")

        x,y = msh.geometry.x[:,0], msh.geometry.x[:,1]

        ux = scalar_to_array(msh, u_dolf.sub(0))
        uz = scalar_to_array(msh, u_dolf.sub(1))
        d = scalar_to_array(msh, w.sub(0))

        connty = msh.topology.connectivity(2, 0)
        connty_array = np.array([connty.links(i) 
                        for i in range(connty.num_nodes)])
        # x_def = x + ux
        # y_def = y + uz
        tess = tri.Triangulation(
                x, 
                y, 
                triangles=connty_array)

        # ax = axs[j,i]
        ax = fig.add_subplot(gs[j, i])
        ax.axis('off')
        ax.set_aspect(aspect=1)
        
        ax.set_xlim(zoom_x)
        ax.set_ylim(zoom_y)
        #axis off
        
        ## plot d
        tpc = ax.tripcolor(
            tess,
            d,
            shading='gouraud',   # smooth shading
            cmap='coolwarm',      # or 'viridis', 'RdYlBu_r', etc.
            vmin=0, vmax=1       # adjust to your field range
            )
        
        # add label in top left corner
        ax.text(0.05, 0.9, '(' +labels[i*len(ls)+j]+')', transform=ax.transAxes, va='top', ha='left')

#titles
        if i == 0 and j == 0:
            ax.set_title(model1)
            # ax.set_title('$∇(σ)')
        elif i == 1 and j == 0:
            ax.set_title(model2)

# add text in between giving l values
        if i == 0:
            ax.text(1.1, 0.5, '$l = ' + str(l) + '$ m', transform=ax.transAxes, va='center', ha='center')



data = np.load('elastic_l' + str(ls[0]) + '_Gc0.5_psicrit1.0.npz', allow_pickle=True)
x = data['x']
y = data['z']
connty_array = data['contty']
d = data['d']

tess = tri.Triangulation(
        x,
        y,
        triangles=connty_array)
# Bottom one spanning both columns
ax = fig.add_subplot(gs[-1, :])
ax.axis('off')
ax.set_aspect(aspect=1)

tpc = ax.tripcolor(
    tess,
    d,
    shading='gouraud',   # smooth shading
    cmap='coolwarm',      # or 'viridis', 'RdYlBu_r', etc.
    vmin=0, vmax=1       # adjust to your field range
    )

# draw box on last plot to indicate zoomed area
rect = plt.Rectangle((zoom_x[0], zoom_y[0]),
                     zoom_x[1]-zoom_x[0],
                     zoom_y[1]-zoom_y[0],
                     linewidth=1,
                     edgecolor='black',
                     facecolor='none',
                     linestyle='--')
ax.add_patch(rect)

ax.text(0.05, 0.8, '(k)', transform=ax.transAxes, va='top', ha='left')

## add floating colorbar below
cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(tpc, cax=cbar_ax, orientation='horizontal')
cbar.set_label('$d$')

plt.tight_layout()
        
plt.savefig('damagecomparison.pdf', bbox_inches='tight')
plt.savefig('damagecomparison.png', bbox_inches='tight', dpi=600)
       