#%%
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

d = sp.symbols('d', positive=True)
g = (lambda d: (1 - d)**2)


l = sp.symbols('l', positive=True)
HH = sp.symbols('H', positive=True)
C3 = sp.symbols('C3', positive=True)

w = lambda d: d
wAT2 = lambda d: d**2

plt.figure()



cw = 4*sp.integrate(sp.sqrt(w(d)), (d,0,1))

gprime = sp.diff(g(d), d)
wprime = sp.diff(w(d), d)

eq = sp.Eq(wprime/(cw*l), -C3*gprime*HH)

sol = sp.solve(eq, d)
d_sol = sp.simplify(sol[0]) 
print(d_sol)



# plot stress strain curve
ε = np.array(np.linspace(1e-5,30,300))

psi = 0.5*ε**2 


d_vals = [d_sol.subs({C3:1, l:0.01, HH:H_val}).evalf() for H_val in psi]

d_vals = np.clip(d_vals, 0, 1)
gvals = (1-np.array(d_vals))**2
σ = gvals*ε

plt.plot(ε, σ)


# plt.ylim(0,20)

# %%
