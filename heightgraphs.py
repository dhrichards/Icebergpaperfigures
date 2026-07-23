#%%
import numpy as np
from matplotlib import pyplot as plt


berg_heights = np.array([300,400,600,550,450,500,350,325,650,700])

berg_dts = 5.0

berg_fail_its = np.array([721,38,24,25,32,29,44,200,20,20])

bergs_fail_times = (berg_fail_its-10)*berg_dts

cliff_heights = [70, 75 ,80]

cliff_dts = [0.1, 0.1, 0.1]

cliff_fail_its = [1000,4,2 ]


## plot iceberg time to failure vs height
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.scatter(berg_heights, bergs_fail_times, marker='o', label='Icebergs')

ax.set_yscale('log')
ax.set_xscale('log')

#polyfit for height>340
mask = berg_heights<340
coeffs = np.polyfit(np.log(berg_heights[mask]), np.log(bergs_fail_times[mask]), 1)
print('Polyfit coefficients for iceberg time to failure vs height:', coeffs)
x_fit = np.linspace(300, 700, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
ax.plot(x_fit, y_fit, label=f'Fit: time ~ height^{coeffs[0]:.2f}', ls='--') 
ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

ax.legend()
ax.set_title('Time to failure vs height for icebergs')


