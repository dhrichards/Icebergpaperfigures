#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

df = pd.read_csv("calvingareas/calving_results.csv")

df['calving_rate'] = df['calved_area']*df['H']/df['t_fail_days']
df['lstar'] = 2*df['lstar']
df['cellfactor'] = 0.5/df['cellfactor']
parameter_labels = {
    'L': (r'L/H', ''),
    'H': (r'H', r'\mathrm{m}'),
    'lstar': (r'\ell^*', ''),
    'dtstar': (r'\Delta t^*', ''),
    'strength': (r'\sigma_t', r'\mathrm{kPa}'),
    'Kic': (r'K_{IC}', r'\mathrm{kPa\,m^{1/2}}'),
    'cellfactor': (r'h/\ell^*', ''),
    'n': (r'n', ''),
}
df = df[
    (df["strength"] == 150) |
    (df["Kic"] == 100)
]
E = 9.33e9; ν=0.325
μ = E/(2*(1+ν))
psicrit = df['strength']**2/(2*E)
Hc = (0.1*900*9.8*df['H'])**2/μ

df['psicritstar'] = psicrit/Hc

def get_markers(group_dict):
    """Return a list of marker(s) to overlay for this group."""
    strength_150 = group_dict.get('strength') == 150
    kic_100 = group_dict.get('Kic') == 100

    if strength_150 and kic_100:
        return ['X', 'P']    # overlay both
    elif strength_150:
        return ['P']
    elif kic_100:
        return ['X']
    else:
        return ['o']


def plot_data(
    ax,
    data,
    x,
    y='t_fail_days',
    group_by=['L','lstar','dtstar','strength','Kic','cellfactor','n'],
    powerlaw=True,
    filters=None,
    loglog=True,
    minlength=2,
    sublegend=True,          # NEW: add small slope legend to this ax
    sublegend_loc='best',
    sublegend_fontsize='medium',
):
    if filters is not None:
        for parameter, values in filters.items():
            data = data[data[parameter].isin(values)]

    if group_by:
        groups = data.groupby(group_by)
    else:
        groups = [('all', data)]

    handles, labels = [], []        # for the MAIN legend
    slope_handles, slope_labels = [], []   # for the per-axis slope legend

    for group_number, (group_name, group) in enumerate(groups):

        group = group.sort_values(x)
        x_data = group[x].to_numpy()
        y_data = group[y].to_numpy()

        if len(group) < minlength:
            continue

        if group_by:
            if not isinstance(group_name, tuple):
                group_name = (group_name,)
            group_dict = dict(zip(group_by, group_name))
            label_parts = []
            for param, value in zip(group_by, group_name):
                symbol, unit = parameter_labels.get(param, (param, ''))
                if unit:
                    label_parts.append(rf'${symbol}={value:g}\:{unit}$')
                else:
                    label_parts.append(rf'${symbol}={value:g}$')
            label = ', '.join(label_parts)
        else:
            label = None

        markers = get_markers(group_dict)


        if powerlaw:
            coeffs = np.polyfit(np.log(x_data), np.log(y_data), 1)
            p = coeffs[0]
            A = np.exp(coeffs[1])

            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = A * x_fit**p

            line, = ax.plot(x_fit, y_fit, '--')
            color = line.get_color()

            # overlay one scatter per marker, same color, same points
            scatter_group = []
            for m in markers:
                s = ax.scatter(x_data, y_data, marker=m, color=color)
                scatter_group.append(s)

            # combine into single legend entry (tuple = overlaid in legend too)
            handles.append(tuple(scatter_group))
            label += rf', $p={p:.2f}$'
            labels.append(label)

            # slope legend entry instead of annotate
            slope_handles.append(line)
            slope_labels.append(rf'${p:.2f}$')
        else:
            line, = ax.plot(x_data, y_data, marker='o', label=label)
            handles.append(line)
            labels.append(label)

    

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter

        H_ticks = [300, 400, 500, 600, 700, 800, 1000]
        ax.xaxis.set_major_locator(FixedLocator(H_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in H_ticks]))

        # minor ticks between the major ones
        H_minor_ticks = [410,420,430,440,450,460,470,480,490,510,520,530,540,550,560,570,580]
        ax.xaxis.set_minor_locator(FixedLocator(H_minor_ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f'${x}$')

    if powerlaw and sublegend and slope_handles:
        ax.legend(
            slope_handles,
            slope_labels,
            loc=sublegend_loc,
            fontsize=sublegend_fontsize,
            frameon=True,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.25,
            title='Slope',
            title_fontsize=sublegend_fontsize,
        )

    return handles, labels


fig, axes = plt.subplots(2, 2, figsize=(7, 7))

filt = {'dtstar': [10], 'lstar': [0.016], 'cellfactor': [0.5], 'n': [3],
        'strength': [100, 150, 200], 'L': [5],
        'H': [450,500,550]}
group = ['strength', 'Kic']

plot_data(axes[0, 0], df, 'H', 't_fail_days', filters=filt, group_by=group)
main_handles, main_labels = plot_data(
    axes[0, 1], df, 'H', 'calved_area', filters=filt, group_by=group
)
plot_data(axes[1, 0], df, 'H', 'calving_rate', filters=filt, group_by=group)

from matplotlib.ticker import FuncFormatter, NullFormatter
axes[0,1].yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

axes[0, 0].set_ylabel('Time to failure (days)')
axes[0, 1].set_ylabel('Non-dimensional calved distance')
axes[1, 0].set_ylabel('Calving rate (m/day)')

axes[1, 1].legend(
    main_handles,
    main_labels,
    loc='center',
    frameon=False,
)
axes[1, 1].set_axis_off()

import string

subplot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]  # skip axes[1,1], it's the legend panel

for ax, letter in zip(subplot_axes, string.ascii_lowercase):
    ax.text(
        0, 1.1,
        f'({letter})',
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left',
    )

fig.tight_layout()

fig.savefig('calvingrates.pdf',bbox_inches='tight')
#%%
fig,ax = plt.subplots(figsize=(4,4))

filt = {'n': [3], 'strength': [200], 'Kic': [100],'lstar':[0.007,0.008]}
group = ['lstar','cellfactor','dtstar','L']
plot_data(ax,df,'H','calved_area',filters=filt,group_by=group,loglog=False,minlength=1)
ax.legend()

#%%
fig,ax = plt.subplots(figsize=(4,4))

filt = {'n': [3], 'strength': [100], 'Kic': [100]}
group = ['lstar','cellfactor','dtstar','L']
h,l =plot_data(ax,df,'H','calving_rate',
          filters=filt,group_by=group,loglog=True,minlength=1,sublegend=True)

    
ax.set_ylabel(f'Calving rate (m/day)')
ax.legend(h,l,loc='center left', bbox_to_anchor=(1,0.5))
# ax.grid(which='both')

# fig.tight_layout()
fig.savefig('meshindependence.pdf',bbox_inches='tight')
