#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
# use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 12
})

# heights = [250,300,350,400,450]
# fail_its = [137,40,18,9,6]

# dt = 1.0
# fail_times = (np.array(fail_its))*dt

heightsar20 = [275,300,350,400]
fail_its2 = [58,24,4,1]
dt2 = 0.25
fail_times2 = (np.array(fail_its2))*dt2




heightsar10dt025 = [250,300,350,400,450]
fail_its_ar10 = [488,138,59,28,14]
fail_times_ar10 = (np.array(fail_its_ar10))*dt2
fail_length_ar10 = 5*np.ones_like(heightsar10dt025)

heightsar40 = [300,350,375]
fail_its_ar40 = [51,9,5]
fail_times_ar40 = (np.array(fail_its_ar40))*dt2
fail_length_ar40_l = [0,10.2,10.2]
fail_length_ar40_r = [0,29,29]

fail_length_ar40 = 0.5*(np.array(fail_length_ar40_l) + (40-np.array(fail_length_ar40_r)))


heightsar30 = [285,300,350,375]
fail_its_ar30 = [100,46,8,4]
fail_times_ar30 = (np.array(fail_its_ar30))*dt2


heightsar30l04 = [275]
fail_its_ar30l04 = [52]



fig, axs = plt.subplots(1,2,figsize=(8,4))

ax = axs[0]
# ticks as 200, 300, not 2x10^2, 3x10^2
ax.scatter(heightsar10dt025, fail_times_ar10, marker='o', label='$L = 10H$')
ax.scatter(heightsar20, fail_times2, marker='s', label='$L = 20H$')
ax.scatter(heightsar40, fail_times_ar40, marker='^', label='$L = 40H$')
ax.scatter(heightsar30, fail_times_ar30, marker='d', label='$L = 30H$')
ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')
# ax.set_title('Iceberg time to failure vs height', fontsize=16)



ax.set_yscale('log')
ax.set_xscale('log')


for h,t in zip([heightsar10dt025, heightsar20, heightsar40, heightsar30],
             [fail_times_ar10, fail_times2, fail_times_ar40, fail_times_ar30]):
    coeffs = np.polyfit(np.log(h), np.log(t), 1)
    x_fit = np.linspace(min(h)-50, max(h)+50, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$')

#legend outside

# ax.legend(loc='upper left', bbox_to_anchor=(1, 1),ncol=2)


from matplotlib.ticker import FuncFormatter

ticks = [200, 300, 400, 500]

ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f'{y:g}')
)


# axs[1].plot([0,50], [0,25], ls='--', color='gray', label=r'$L_{frac}=0.5L$')
axs[1].scatter(heightsar40, fail_length_ar40, marker='^', label='$L = 40H$')
axs[1].scatter(heightsar10dt025, fail_length_ar10, marker='o', label='$L = 10H$')
axs[1].set_xlabel('$L/H$')
axs[1].set_ylabel(r'$L_{frac}/H$')

# no minor ticks
ax.grid(True, which='major', linestyle='--', alpha=0.5)

fig.savefig('iceberg_time_to_failure_vs_height.png', dpi=300, bbox_inches='tight')
fig.savefig('iceberg_time_to_failure_vs_height.pdf',bbox_inches='tight')



# %%


