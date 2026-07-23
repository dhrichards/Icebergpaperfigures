#%%
import numpy as np
from matplotlib import pyplot as plt
import cmasher as cmr
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 12
})



E = 9.33e9
Gc = 1.0
σc = 200e3

ψcrit = σc**2/(2*E)
εc = σc/E

ε = np.linspace(0,3*εc,1000)

lcrit = 3*E*Gc/(16/3*σc)**2

ψ0 = 0.5*E*ε**2


cmap = cmr.get_sub_cmap('cmr.freeze_r',0,1)
def g_lo(d,s):
    ϕ = 1-d
    return s*(1-((s-1)/s)**(ϕ**2))

fig, axs = plt.subplots(1,2,figsize=(6,3),sharey=True)

for ψc, ax in [ (0, axs[0]), (ψcrit, axs[1]) ]:
    H = ψ0 - ψc
    H[H<0] = 0
    for l_factor in [1,10,100]:
        l = l_factor*lcrit
        d = 2*H*l/(Gc + 2*H*l)
        g = (1-d)**2
        # g = g_lo(d,1 + 1e-15)
        σ = g*E*ε

        ax.plot(ε/εc, σ/σc, label=r'$\ell = {} \ell_{{crit}}$'.format(l_factor))
        # ax.plot(ε/εc, d)

for ax in axs:
    ax.set_xlabel(r'$\varepsilon/\varepsilon_t$')
    # x ticks every 1
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid()
    ax.set_ylim(0,1.2)
    ax.set_xlim(0,3)

axs[0].set_ylabel(r'$\sigma/\sigma_t$')
# joint legend outside, below
axs[1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), ncol=3)

axs[0].set_title(r'(a) $\psi_{crit} = 0$')
axs[1].set_title(r'(b) $\psi_{crit} = \sigma_{t}^2/(2E)$')

fig.savefig('1dphasefield.png', bbox_inches='tight', dpi=400)
fig.savefig('1dphasefield.pdf', bbox_inches='tight')
