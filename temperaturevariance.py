#%%
import numpy as np
from matplotlib import pyplot as plt


Ts = -20; Tb = -1

z = np.linspace(-0.9, 0.1, 100) # z

T = -(Tb - Ts)*z + Ts

ft0 = 0.2e6
ftdeg = 0.04e6
ft = ft0 - ftdeg*T


fig, ax = plt.subplots()
ax.plot(ft,z)
ax.set_xlabel('Tensile strength (Pa)')
ax.set_ylabel('Z (m)')

E = 9.33e9
ψcrit = ft**2/(2*E)
fig2, ax2 = plt.subplots()
ax2.plot(ψcrit,z)
ax2.set_xlabel('\psi_crit')
ax2.set_ylabel('Z (m)')

l=2
Gc = 8*ft**2*l/(3*E)
fig3, ax3 = plt.subplots()
ax3.plot(Gc,z)
ax3.set_xlabel('Gc (J/m^2)')
ax3.set_ylabel('Z (m)')

# ν = 0.325
# Kic = 
# Gc = Kic**2*(1-ν**2)/E