#%%
import numpy as np
import pandas as pd

rows = []

def add_dataset(H, its, dt=0.1, l=8, L=None, aspect_ratio=None, T_variation=True, label=None):
    for h, it in zip(H, its):
        rows.append({
            "H": h,
            "its": it,
            "time": it * dt,
            "dt": dt,
            "l": l,
            "L": L,
            "aspect_ratio": aspect_ratio,
            "T_variation": T_variation,
            "label": label
        })


add_dataset(
    H = np.array([325,350,400,425,450,475,500,525,550,575]),
    its = np.array([121,129,54,48,31,29,19,15,17,16]),
    L = 4000
    )

add_dataset(
    H = np.array([350,400,450,500,550]),
    its = np.array([34,19,13,11,14]),
    L = 4000,
    l = 5
    )

add_dataset(
    H = np.array([250,300,350,400,450,500,550,600,650]),
    its = np.array([475,362,294,284,265,270,264,279,282]),
    L = 2000
    )

add_dataset(
    H = np.array([350,450,500,550]),
    its = np.array([194,58,15,5]),
    L = 8000
    )

add_dataset(
    H = np.array([350,400,450,500,550]),
    its = np.array([418,21,33,25,12]),
    L = 16000
)

add_dataset(
    H = np.array([350,400,450,500]),
    its = np.array([181,86,24,14]),
    L = 8000,
    l = 5
)

add_dataset(
    H = np.array([350,400,450,500,550]),
    its = np.array([498,285,158,103,71]),
    aspect_ratio = 5
)

add_dataset(
    H = np.array([350,400,450,500]),
    its = np.array([175, 55, 23,11]),
    aspect_ratio = 10
)    

add_dataset(
    H = np.array([350,400,450,500,550]),
    its = np.array([39,15,6,3,48]),
    aspect_ratio = 10,
    T_variation = False
)   

add_dataset(
    H = np.array([350,400,450,500,550]),
    its = np.array([98, 68, 18, 42, 38]),
    L = 8000,
    T_variation = False
)

df = pd.DataFrame(rows)
group_cols = ["L", "l", "aspect_ratio", "T_variation"]
groups = df.groupby(group_cols, dropna=False)

def is_valid_group(key, group):
    L, l, ar, T_var = key
    
    # must have at least one defining parameter
    if all(pd.isna([L, l, ar])):
        return False
    
    # need enough data points
    if len(group) < 3:
        return False
    
    return True

from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(6,5))
x_fit = np.linspace(300, 700, 100)

]

for key, group in groups:
    if not is_valid_group(key, group):
        continue

    H = group["H"].values
    time = group["time"].values

    # --- build label dynamically ---
    L, l, ar, T_var = key
    parts = []
    if not pd.isna(L): parts.append(f"L={int(L)}")
    if not pd.isna(l): parts.append(f"l={int(l)}")
    if not pd.isna(ar): parts.append(f"AR={int(ar)}")
    if T_var is False: parts.append("no T")

    label = ", ".join(parts)

    # --- scatter ---
    ax.scatter(H, time, label=label)

    # --- fit ---
    coeffs = np.polyfit(np.log(H), np.log(time), 1)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]

    print(f"{label}: slope = {coeffs[0]:.2f}")

    ax.plot(
        x_fit, y_fit,
        ls="--",
        label=f"{label}: ~ H^{coeffs[0]:.2f}"
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Time to failure (days)")
#legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')