
#%%
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

C3 = sp.symbols('C3', positive=True)
ε = sp.symbols('epsilon', positive=True)
ψcrit = sp.symbols('psi_crit', positive=True)
ν = sp.symbols('nu', positive=True)
E = sp.symbols('E', positive=True)
Gc = sp.symbols('Gc', positive=True)
l = sp.symbols('l', positive=True)



def positive_part(x):
    return (x + sp.Abs(x))/2


## phase field equation
d = sp.symbols('d', positive=True)
g = (lambda d: (1 - d)**2)
w = lambda d: d**2

cw = 4*sp.integrate(sp.sqrt(w(d)), (d,0,1))

gprime = sp.diff(g(d), d)
wprime = sp.diff(w(d), d)

# ψ = 0.5*σ**2 / E
ψ0 = 0.5*E*ε**2

# H = positive_part(ψ0 - ψcrit)
# H = ψ0-ψcrit
H = sp.symbols('H', positive=True)

eq = sp.Eq(Gc*wprime/(cw*l), -gprime*H)
# eq = sp.Eq(Gc*d/l, 2*(1-d)*H)
sol = sp.solve(eq, d)
# print(sol)
d_sol = sp.simplify(sol[0])


# Assert that d_sol>0
# d_sol = 0.5*(d_sol + sp.Abs(d_sol))


σ0 = E * ε
σ = g(d_sol)* E * ε
#%%
# Find intersection between σ and σ0
eq_stress = sp.Eq(σ, σ0)
ε_crit = sp.solve(eq_stress, ε)

σ_crit = []
for val in ε_crit:
    σ_crit.append(σ0.subs(ε, val).evalf())




#%%

#Find maximum of σ through differentiation
dσ_dε = sp.diff(σ, ε)
critical_points = sp.solve(dσ_dε, ε)
# Filter only real positive critical points
critical_points = [cp.evalf() for cp in critical_points if cp.is_real and cp > 0]
# Find maximum stress
max_stress = None
if critical_points:
    ε_crit = critical_points[0]
    max_stress = σ.subs(ε, ε_crit).evalf()
    # print(f'Maximum stress: {max_stress} at strain: {ε_crit}')

#%%
# plot stress strain curve
ε_vals = np.linspace(0,20,300)
Eval = 1.0
Gcval = 1.0
ψcritval = 0.0


fig, ax = plt.subplots()
σ0_vals = [σ0.subs({E:Eval, ε:eps}).evalf() for eps in ε_vals]
ax.plot(ε_vals[1:], σ0_vals[1:], '--')
for lval in [0.001,0.002]:

    σ_vals = [σ.subs({E:Eval, Gc:Gcval, l:lval, ε:eps, ψcrit:ψcritval}).evalf() for eps in ε_vals]
    


    ax.plot(ε_vals[1:], σ_vals[1:])

# # plot critical stresses as horizontal lines
# σcrit_vals = [sc.subs({E:Eval, Gc:Gcval, l:lval, ψcrit:ψcritval}).evalf() for sc in σ_crit]
# for sc in σcrit_vals:
#     ax.axhline(y=sc, color='r', linestyle='--')
# ax.set_ylim(0,5)
# %%
