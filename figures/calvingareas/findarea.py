#%%
import adios4dolfinx
import kraken as kr
import kraken.numerics.energy_splits as es
import kraken.numerics.maths_functions as mf
import numpy as np
import ufl
from matplotlib import pyplot as plt
from matplotlib import tri
from mpi4py import MPI
import dolfinx
from collections import defaultdict, deque

def find_calved_region(tess, d, threshold=0.5):

    triangles = tess.triangles
    x = tess.x
    y = tess.y
    ntri = len(triangles)

    # d values in each triangle
    d_tri = d[triangles]

    # Treat triangles containing the fracture as barriers
    fractured = np.max(d_tri, axis=1) > threshold

    # ---------------------------------------------------------
    # Build triangle adjacency
    # ---------------------------------------------------------
    edge_to_tri = defaultdict(list)

    for i, tri in enumerate(triangles):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]

        for edge in edges:
            edge_to_tri[edge].append(i)

    # Full adjacency: all neighbouring triangles
    adjacency_full = [[] for _ in range(ntri)]

    for tris in edge_to_tri.values():
        if len(tris) == 2:
            a, b = tris
            adjacency_full[a].append(b)
            adjacency_full[b].append(a)

    # Restricted adjacency for main-ice flood fill
    adjacency = [[] for _ in range(ntri)]

    for t in range(ntri):
        if fractured[t]:
            continue

        for neighbour in adjacency_full[t]:
            if not fractured[neighbour]:
                adjacency[t].append(neighbour)

    # ---------------------------------------------------------
    # Seed flood fill from triangles at the left-hand side
    # ---------------------------------------------------------

    cx = np.mean(x[triangles], axis=1)

    # Use the leftmost 1% of triangles
    n_seed = max(1, int(0.01 * ntri))

    seed_order = np.argsort(cx)
    seeds = seed_order[:n_seed]

    # Remove fractured seeds
    seeds = seeds[~fractured[seeds]]

    # ---------------------------------------------------------
    # Flood fill main ice
    # ---------------------------------------------------------

    connected = np.zeros(ntri, dtype=bool)

    queue = deque(seeds)

    for s in seeds:
        connected[s] = True

    while queue:
        t = queue.popleft()

        for neighbour in adjacency[t]:
            if not connected[neighbour]:
                connected[neighbour] = True
                queue.append(neighbour)

    # Anything not connected to the main body is calved.
    # Exclude the fracture itself.
    calved = ~connected & ~fractured

    # ---------------------------------------------------------
    # Find the ENTIRE connected damaged region adjacent to
    # the calved region
    # ---------------------------------------------------------

    adjacent_damage = np.zeros(ntri, dtype=bool)

    # Fractured triangles directly touching the calved region
    damage_seeds = set()

    for t in np.where(calved)[0]:
        for neighbour in adjacency_full[t]:
            if fractured[neighbour]:
                damage_seeds.add(neighbour)

    damage_seeds = np.array(list(damage_seeds), dtype=int)

    # Flood fill through all connected fractured triangles
    queue = deque(damage_seeds)

    for s in damage_seeds:
        adjacent_damage[s] = True

    while queue:
        t = queue.popleft()

        for neighbour in adjacency_full[t]:
            if fractured[neighbour] and not adjacent_damage[neighbour]:
                adjacent_damage[neighbour] = True
                queue.append(neighbour)

    # ---------------------------------------------------------
    # Calculate area
    # ---------------------------------------------------------

    p0 = np.column_stack((
        x[triangles[:, 0]],
        y[triangles[:, 0]]
    ))

    p1 = np.column_stack((
        x[triangles[:, 1]],
        y[triangles[:, 1]]
    ))

    p2 = np.column_stack((
        x[triangles[:, 2]],
        y[triangles[:, 2]]
    ))

    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )

    calved_area = areas[calved].sum()
    adjacent_damage_area = areas[adjacent_damage].sum()

    return (
        calved,
        adjacent_damage,
        fractured,
        calved_area,
        adjacent_damage_area,
    )
#find folders in current directory
import os
folders = [f for f in os.listdir('.') if os.path.isdir(f)]

results = []
for folder in folders:
    # extract attributes fromf folder
    L = float(folder.split('L')[1].split('_')[0])
    H = float(folder.split('H')[1].split('_')[0])
    lstar = float(folder.split('_l')[1].split('_')[0])
    dtstar = float(folder.split('dtstar')[1].split('_')[0])
    strength = float(folder.split('sigmac')[1].split('_')[0])
    level = float(folder.split('level')[1].split('_')[0])
    Kic = float(folder.split('Kic')[1].split('_')[0])
    cellfactor = float(folder.split('cellfactor')[1].split('_')[0])
    T = float(folder.split('T')[1].split('_')[0])
    lfactor = float(folder.split('_lfactor')[1].split('_')[0])
    n = float(folder.split('_n')[1].split('_')[0])

    t = adios4dolfinx.read_timestamps(folder, MPI.COMM_WORLD, function_name = "w_damage")

    msh = adios4dolfinx.read_mesh(folder, MPI.COMM_WORLD, time=t[-1])
    model = kr.base.Simulation(msh)
    model.basal_friction = False

    x = ufl.SpatialCoordinate(msh)
    z = x[msh.geometry.dim-1]
    model.params.T.value = -T
    model.params.A0.value = mf.rate_factor_np(T)
    model.params.H.value = H
    model.params.l.value = lstar*H
    model.params.Kic.value = Kic*1e3
    model.params.patm.value = 0.0
    model.params.crack_level_above_sea.value = level
    model.params.ρc = dolfinx.fem.Constant(model.msh,0.1*900)

    model.params.dt.value = dtstar*model.params.τ_float
    model.params.σc = strength*1e3 

    
    
    
    model.setup()
    model.read_checkpoint(folder, t=t[-1])

    t_fail_days = t[-1]/(24*3600)
    

    adios4dolfinx.read_function(folder, model.momentum.w, name ="w_momentum", time=t[-1])
    
    tess = kr.plotting.get_triangulation(msh)

    d = kr.plotting.dolfinx_to_array(msh,model.damage.d)    

    threshold = 0.3

    calved, adjacent, fractured, calved_area, adjacent_area = find_calved_region(
        tess,
        d,
        threshold
    )
    area = calved_area + adjacent_area

    results.append({
        "folder": folder,
        "L": L,
        "H": H,
        "lstar": lstar,
        "dtstar": dtstar,
        "strength": strength,
        "level": level,
        "Kic": Kic,
        "cellfactor": cellfactor,
        "T": T,
        "lfactor": lfactor,
        "n": n,
        "t_fail_days": t_fail_days,
        "calved_area": area,
        "area_threshold": threshold,
    })



    print(f"Calved area = {area:.2f}")

    plt.figure(figsize=(10, 5))

    # Plot d
    # plt.tripcolor(
    #     tess,
    #     d,
    #     shading="gouraud"
    # )

    # Overlay calved triangles
    plt.tripcolor(
        tess.x,
        tess.y,
        tess.triangles[calved],
        facecolors=np.ones(np.sum(calved)),
        edgecolors="red",
        alpha=0.5
    )

    # Overlay adjacent damage
    plt.tripcolor(
        tess,
        facecolors=np.where(adjacent, 1.0, np.nan),
        cmap="Greys",
        alpha=0.5,
    )


    plt.axis("equal")
    plt.colorbar(label="d")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(folder)

    plt.show()


# -------------------------------------------------------------
# Save everything to CSV
# -------------------------------------------------------------
import pandas as pd
df = pd.DataFrame(results)

df.to_csv("calving_results.csv", index=False)

print(df)
print(f"\nSaved {len(df)} runs to calving_results.csv")