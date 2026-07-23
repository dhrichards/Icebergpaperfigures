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

dt=0.1


heightsar40 = [300,350,375]
fail_its_ar40 = [162,11,5]
fail_t_ar40 = (np.array(fail_its_ar40))*dt
fail_dist_ar40 = [9.4,6.1,12.4]

heightsar20 = [300,325,350,375,400]
fail_its_ar20 = [68,29,2,2,1]
fail_t_ar20 = (np.array(fail_its_ar20))*dt
fail_dist_ar20 = [10,5.6,np.nan,10]

heightsar30 = [350,375]
fail_its_ar30 = [13,2]
fail_t_ar30 = (np.array(fail_its_ar30))*dt





fig, ax = plt.subplots(1,1,figsize=(4,4))

# ax = axs[0]
# ticks as 200, 300, not 2x10^2, 3x10^2
ax.scatter(heightsar20, fail_t_ar20, marker='s', label='$L = 20H$')
ax.scatter(heightsar40, fail_t_ar40, marker='^', label='$L = 40H$')
ax.scatter(heightsar30, fail_t_ar30, marker='o', label='$L = 30H$')
ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')
# ax.set_title('Iceberg time to failure vs height', fontsize=16)



ax.set_yscale('log')
ax.set_xscale('log')


for h,t in zip([heightsar20, heightsar40, heightsar30],
             [fail_t_ar20, fail_t_ar40, fail_t_ar30]):
    coeffs = np.polyfit(np.log(h), np.log(t), 1)
    x_fit = np.linspace(min(h)-50, max(h)+50, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$')

#legend outside

ax.legend(loc='upper left',ncol=2)#, bbox_to_anchor=(1, 1),ncol=2)


from matplotlib.ticker import FuncFormatter

ticks = [200, 300, 400, 500]

ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f'{y:g}')
)


# axs[1].plot([0,50], [0,25], ls='--', color='gray', label=r'$L_{frac}=0.5L$')
# axs[1].scatter(40*np.ones_like(fail_length_ar40), fail_length_ar40, marker='^', label='$L = 40H$')
# axs[1].scatter(10*np.ones_like(fail_length_ar10), fail_length_ar10, marker='o', label='$L = 10H$')
# axs[1].set_xlabel('$L/H$')
# axs[1].set_ylabel(r'$L_{frac}/H$')

# no minor ticks
ax.grid(True, which='major', linestyle='--', alpha=0.5)

fig.savefig('iceberg_time_to_failure_vs_height.png', dpi=300, bbox_inches='tight')
fig.savefig('iceberg_time_to_failure_vs_height.pdf',bbox_inches='tight')



# %%


