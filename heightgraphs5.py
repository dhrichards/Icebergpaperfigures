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

dt=2.5


# heights = [450,500,550]
# fail_its = [611,228,85]

heights=  [350,400,450,500,550]
fail_its = [620,332,193,113,69]
fail_t = (np.array(fail_its))*dt


heightsdt5 = [400,500,550]
fail_itsdt5 = [176,61,39]
fail_tdt5 = (np.array(fail_itsdt5))*5

heightsl0075 = [400,500,550]
fail_itsl0075 = [325,113,70]
fail_tl0075 = (np.array(fail_itsl0075))*dt

Kics = [75,100, 150, 200]
fail_itsKics = [84,113, 171, 228]
fail_tKics = (np.array(fail_itsKics))*dt


strengths = [100,150,200,250,300]
fail_itsstrengths = [69,91,113,134,157]
fail_tstrengths = (np.array(fail_itsstrengths))*dt

heightsL10 = [400,450,510,550]
fail_itsL10 = [334,197,108,88]
fail_tL10 = (np.array(fail_itsL10))*dt


heights_nosmooth = [400,450,500,550]
fail_its_nosmooth = [365,216,129,81]
fail_t_nosmooth = (np.array(fail_its_nosmooth))*dt


heights_divuvnew = [300,350,400,450,500,550]
fail_its_divuvnew = [1119,543,304,178,109,67]
fail_t_divuvnew = (np.array(fail_its_divuvnew))*dt

fig, ax = plt.subplots(1,1,figsize=(4,4))

for h,t,label in zip([heights, heightsL10, heightsdt5, heightsl0075, heights_nosmooth, heights_divuvnew],
             [fail_t, fail_tL10, fail_tdt5, fail_tl0075, fail_t_nosmooth, fail_t_divuvnew],
             ['Runs', 'Runs with L=10', 'Runs with dt=5', 'Runs with l=0.0075', 'Runs with no smoothing', 'Runs with divUvNew']):
    ax.scatter(h, t, marker='o',label=label)
    coeffs = np.polyfit(np.log(h), np.log(t), 1)
    x_fit = np.linspace(min(h)-50, max(h)+50, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$')


ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

# ax.set_title('Iceberg time to failure vs height', fontsize=16)

ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()
#legend outside
ax.legend(ncol=2,bbox_to_anchor=(1, 1))






#%%


plt.figure()
plt.scatter(Kics, fail_tKics, marker='o', label='Runs with varying $K_{Ic}$')
coeffs = np.polyfit(np.log(Kics), np.log(fail_tKics), 1)
x_fit = np.linspace(min(Kics)*0.8, max(Kics)*1.2, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
plt.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto K_{{Ic}}^{{{coeffs[0]:.2f}}}$')
plt.legend()

plt.figure()
plt.scatter(strengths, fail_tstrengths, marker='o', label='Runs with varying strength')
coeffs = np.polyfit(np.log(strengths), np.log(fail_tstrengths), 1)
x_fit = np.linspace(min(strengths)*0.8, max(strengths)*1.2, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
plt.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto strength^{{{coeffs[0]:.2f}}}$')
plt.legend()

ν = 0.325
E = 9.33e9
μ = E/(2*(1+ν))
ρ = 900
g = 9.8
n=3.0
A = 1e-25

def τ(H):
    return (ρ*g*H)**(1-n)/(A*μ)


def C(H,K):
    return ρ**2*g**2*H**3/(K**2)*((2*(1+ν))/(1-ν**2))
Gc = (np.array(Kics)*1e3)**2*(1-ν**2)/E


H_kic = 500
C_Kic = C(H_kic, np.array(Kics)*1e3)


Kic_H = 100e3
Gc_H = (Kic_H)**2*(1-ν**2)/E

C_H = C(np.array(heights), Kic_H)

t_nondim_H = fail_t/τ(np.array(heights))
t_nondim_Kic = fail_tKics/τ(H_kic)

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.scatter(C_H, t_nondim_H, marker='o', label='Runs')
ax.scatter(C_Kic, t_nondim_Kic, marker='s', label='Runs with varying $K_{Ic}$')


ax.set_xscale('log')
ax.set_yscale('log')

#do best fit line for both
coeffs = np.polyfit(np.log(C_H), np.log(t_nondim_H), 1)
x_fit = np.linspace(min(C_H)*0.8, max(C_H)*1.2, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto C^{{{coeffs[0]:.2f}}}$')

coeffs = np.polyfit(np.log(C_Kic), np.log(t_nondim_Kic), 1)
x_fit = np.linspace(min(C_Kic)*0.8, max(C_Kic)*1.2, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto C^{{{coeffs[0]:.2f}}}$')

ax.legend(ncol=2)

# %%



fig, ax = plt.subplots(1,1,figsize=(4,4))

ax.scatter(heights_divuvnew, fail_t_divuvnew, marker='o', label='Runs')

ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')
for h,t in zip([heights_divuvnew],
             [fail_t_divuvnew]):
    coeffs = np.polyfit(np.log(h), np.log(t), 1)
    x_fit = np.linspace(min(h)-10, max(h)+50, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$')

# x_fit = np.linspace(min(heights_divuvnew)-50, max(heights_divuvnew)+50, 100)

# # just fit to y = Ax^n
# n = -5
# A = fail_t[0]/heights[0]**n
# y_fit = A * x_fit**n
# ax.plot(x_fit, y_fit, ls='--', label=rf'$t \propto H^{{{n}}}$')

ax.legend(ncol=2)#, bbox_to_anchor=(1, 1),ncol=2)

fig.savefig('iceberg_time_to_failure_vs_height.png', dpi=300, bbox_inches='tight')
fig.savefig('iceberg_time_to_failure_vs_height.pdf',bbox_inches='tight')
