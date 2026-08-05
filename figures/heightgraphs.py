#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import kraken.numerics.maths_functions as mf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import norm


# use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 12
})


ρi = 900
ρw = 1000
g = 9.8
δ = 1 - ρi/ρw
E = 9.33e9
ν = 0.325
μ = E/(2*(1+ν))
H0 = 450
dt=2.5

def plot(x,its,label=None):
    t = np.array(its)*dt
    plt.figure()
    plt.scatter(x, t, marker='o')

    coeffs = np.polyfit(np.log(x), np.log(t), 1)
    x_fit = np.linspace(min(x)*0.8, max(x)*1.2, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    plt.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto {label}^{{{coeffs[0]:.2f}}}$')
    plt.xlabel(label)
    plt.ylabel('t (days)')
    plt.legend()


def height_plot(ax,h,t,n):
    h = np.array(h)
    label = 'n=' + str(n)
    points = ax.scatter(h, t, marker='o',label=label)
    coeffs = np.polyfit(np.log(h), np.log(t), 1)
    x_fit = np.linspace(min(h)-50, max(h)+50, 100)
    y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
    ax.plot(x_fit, y_fit, ls='--',
             label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$',
             color=points.get_facecolor()[0])


def shifted_power(H, A, H0, n):
    return A * (H + H0)**n

def prescribed_power(H,A,n):
    return A * H**(-n-1)

def prescribed_plot(ax,h,t,n):
    label = 'n=' + str(n)
    points = ax.scatter(h, t, marker='o',label=label)


    # Initial guess
    p0 = [1e12]

    # Nonlinear least-squares fit
    popt, pcov = curve_fit(
        lambda H, A: prescribed_power(H,A,n),
        h,
        t,
        p0=p0,
        maxfev=10000
    )

    A, = popt

    x_fit = np.linspace(min(h)-10, max(h)+50, 300)
    y_fit = prescribed_power(x_fit, A, n)


    # Plot fit
    ax.plot(
        x_fit,
        y_fit,
        '--',
        lw=1,
        color = points.get_facecolor()[0],
        # label=(
        #     rf'$t \propto (H + {H0:.2f})^{{{exponent:.2f}}}$'
        # )
    )


def prescribed_power(H,A,n):
    return A * H**(-n-1)

def prescribed_plot(ax,h,t,n):
    label = 'n=' + str(n)
    points = ax.scatter(h, t, marker='o',label=label)


    # Initial guess
    p0 = [1e12]

    # Nonlinear least-squares fit
    popt, pcov = curve_fit(
        lambda H, A: prescribed_power(H,A,n),
        h,
        t,
        p0=p0,
        maxfev=10000
    )

    A, = popt

    x_fit = np.linspace(min(h)-10, max(h)+50, 300)
    y_fit = prescribed_power(x_fit, A, n)


    # Plot fit
    ax.plot(
        x_fit,
        y_fit,
        '--',
        lw=1,
        color = points.get_facecolor()[0],
        # label=(
        #     rf'$t \propto (H + {H0:.2f})^{{{exponent:.2f}}}$'
        # )
    )


def prescribed_power(H,A,n):
    return A * H**(-n-1)

def prescribed_plot(ax,h,t,n):

    t = t[h<501]
    h = h[h<501]
    label = 'n=' + str(n)
    points = ax.scatter(h, t, marker='o',label=label)


    # Initial guess
    p0 = [1e12]

    # Nonlinear least-squares fit
    popt, pcov = curve_fit(
        lambda H, A: prescribed_power(H,A,n),
        h,
        t,
        p0=p0,
        maxfev=10000
    )

    A, = popt

    x_fit = np.linspace(min(h)-10, max(h)+50, 300)
    y_fit = prescribed_power(x_fit, A, n)


    # Plot fit
    ax.plot(
        x_fit,
        y_fit,
        '--',
        lw=1,
        color = points.get_facecolor()[0],
        label=(
            r'$t \propto H^{' + str(n+1) + '}$' 
        )
    )

    sigma_A = np.sqrt(pcov[0, 0])

    

    # Standard error of A
    sigma_A = np.sqrt(pcov[0, 0])

    # Standard error of fitted curve
    sigma_y = x_fit**(-n-1) * sigma_A

    # 95% confidence interval
    z = norm.ppf(0.975)   # = 1.96
    lower = y_fit - z * sigma_y
    upper = y_fit + z * sigma_y

    # ax.fill_between(
    #     x_fit,
    #     lower,
    #     upper,
    #     color=points.get_facecolor()[0],
    #     alpha=0.25,
    #     linewidth=0,
    # )




def confidence_plot(ax,h,t,n):
    t = t[h>349]
    h = h[h>349]
    t = t[h<551]
    h = h[h<551]
  
    h = np.array(h)
    label = '$n=' + str(n) +'$'
    points = ax.scatter(h, t, marker='o',label=label)

    # Fit in log-log space
    x = np.log(h)
    y = np.log(t)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept, exponent = model.params
    exp_lo, exp_hi = model.conf_int(alpha=0.05)[1]
    intercept_lo, intercept_hi = model.conf_int(alpha=0.05)[0]

    # Smooth curve
    x_fit = np.linspace(min(h)-10, max(h)+50, 300)
    xlog_fit = np.log(x_fit)

    # Best-fit curve
    y_fit = np.exp(intercept + exponent*xlog_fit)

    # Sample joint distribution of (intercept, exponent)
    cov = model.cov_params()
    samples = np.random.multivariate_normal(
        mean=model.params,
        cov=cov,
        size=10000
    )

    # Evaluate sampled curves
    logy_samples = (
        samples[:, 0, None]
        + samples[:, 1, None] * xlog_fit[None, :]
    )

    y_samples = np.exp(logy_samples)

    # Pointwise 95% confidence band
    y_lower = np.percentile(y_samples, 2.5, axis=0)
    y_upper = np.percentile(y_samples, 97.5, axis=0)
    # y_lower = np.exp(intercept_hi + exp_lo*xlog_fit)
    # y_upper = np.exp(intercept_lo + exp_hi*xlog_fit)

    # Plot fit
    # line, = ax.plot(
    #     x_fit,
    #     y_fit,
    #     '--',
    #     lw=1,
    #     label=(
    #         rf'$t \propto H^{{{exponent:.2f}}}$'
    #         '\n'
    #         rf'95\% CI: [{exp_lo:.2f}, {exp_hi:.2f}]'
    #     )
    # )

    # Confidence band
    ax.fill_between(
        x_fit,
        y_lower,
        y_upper,
        color=points.get_facecolor()[0],
        alpha=0.2,
        label = rf'$t \propto H^{{{exp_hi:.2f}}}$ to $H^{{{exp_lo:.2f}}}$'
        # label = rf'95\% CI: $t \propto H^{{{exp_lo:.2f}}}$ to $H^{{{exp_hi:.2f}}}$'
    )





heights_norelax = np.array([300,350,400,450,500,550],dtype=np.float64)
fail_its_norelax = np.array([528,279,170,114,83,64])

heights_nosmooth = np.array([250,300,350,400,450,500,550,600], dtype=np.float64)
its_nosmooth = np.array([1363,588,311,188,127,92,71,58], dtype=np.float64)

h_n2_nosmooth = np.array([250,300,350,400,450,500,550,600], dtype=np.float64)
its_n2_nosmooth = np.array([571,290,172,116,86,68,56,49], dtype=np.float64)

heights_linear = np.array([400,500,550], dtype=np.float64)
fail_its_linearA = np.array([28,22,20])
fail_its_linear = np.array([75,52,45], dtype=np.float64)

heights_lin_nosmooth = np.array([250,300,350,400,450,500,550,600], dtype=np.float64)
its_lin_nosmooth = np.array([291,174,116,86,69,58,52,45], dtype=np.float64)


h_lin_nosmooth_L10 = np.array([300,500], dtype=np.float64)
its_lin_nosmooth_L10 = np.array([177,59], dtype=np.float64)

h_nosmooth_L10 = np.array([300,500], dtype=np.float64)
its_nosmooth_L10 = np.array([597,92], dtype=np.float64)

heights_n2 = np.array([350,450,550], dtype=np.float64)
its_n2 = np.array([83,46,32], dtype=np.float64)


h_n3_dstar10 = np.array([350,400,450,500,550], dtype=np.float64)
t_n3_dstar10 = np.array([762.09,441.869,284.694,201.658,148.62]) 

h_n3_dtstar10_l5 = np.array([500])
t_n3_dtstar10_l5 = np.array(196.439)

h_n1_dstar10 = np.array([350,400,450,500,550], dtype=np.float64)
t_n1_dstar10 = np.array([290.98,211.1,167.36,138.83,117.913])

h_n1_dtstar10_l5 = np.array([500],dtype=np.float64)
t_n1_dtstar10_l5 = np.array([135.03])

h_n2_dstar10 = np.array([350,400,450,500,550], dtype=np.float64)
t_n2_dstar10 = np.array([422.05,275.485,196.32,147.24,116.584])

h_n3_dstar1 = np.array([350,450,550], dtype=np.float64)
t_n3_dstar1 = np.array([59101083,22355439,11184098])/(24*60*60)

h_n1_dstar1 = np.array([350,450,550], dtype=np.float64)
t_n1_dstar1 = np.array([21788644,12225302,8133769])/(24*60*60)

fig, ax = plt.subplots(1,1,figsize=(4,4))



# prescribed_plot(ax,h_n3_dstar10, t_n3_dstar10, 3)
# prescribed_plot(ax,h_n2_dstar10, t_n2_dstar10, 2)
# prescribed_plot(ax,h_n1_dstar10, t_n1_dstar10, 1)

# height_plot(ax,heights_nosmooth,its_nosmooth*2.5,3)
confidence_plot(ax,h_n3_dstar10, t_n3_dstar10, 3)
confidence_plot(ax,h_n2_dstar10, t_n2_dstar10, 2)
confidence_plot(ax,h_n1_dstar10, t_n1_dstar10, 1)

ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

# ax.set_title('Iceberg time to failure vs height', fontsize=16)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
#legend outside
ax.legend(ncol=1,bbox_to_anchor=(1, 1))


#%%

def tau(H,n):
    δ = 0.1; ρi = 900; g=9.8
    E = 9.33e9; ν=0.325
    μ = E/(2*(1+ν))
    A = mf.rate_factor_np(-10)*(0.5*0.1*900*9.8*500)**(3-n)
    τxx = δ*ρi*g*H/4
    τe2 = np.sqrt(τxx**2)
    σc = δ*ρi*g*H/4
    τ = (σc)**(1-n)/(A*μ)
    return τ/(24*60*60)

fig,ax = plt.subplots(1,1,figsize=(4,4))

confidence_plot(ax,h_n3_dstar10, t_n3_dstar10/tau(h_n3_dstar10,3), 3)
confidence_plot(ax,h_n2_dstar10, t_n2_dstar10/tau(h_n2_dstar10,2), 2)
confidence_plot(ax,h_n1_dstar10, t_n1_dstar10/tau(h_n1_dstar10,1), 1)

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(ncol=1,bbox_to_anchor=(1, 1))

#%%
fig,axs = plt.subplots(1,1,figsize=(4,4))

confidence_plot(axs[0],h_n3_dstar10, t_n3_dstar10, 3)
confidence_plot(axs[1],h_n1_dstar10, t_n1_dstar10, 1)
#%%
import kraken as kr
Ts = [-20,-15,-10,-5]
A = kr.numerics.maths_functions.rate_factor_np(np.array(Ts))
fail_its_A = [208,128,83,40]
fail_t_A = np.array(fail_its_A)*dt

plot(A,fail_its_A,'A')

#%%
# Kics = [75,100,200]
Kics = [50,100,200,300]
fail_its_Kic = [125,127,173,256]
fail_t_Kic = np.array(fail_its_Kic)*dt

plot(Kics,fail_its_Kic,'Kic')
#%%
strengths = np.array([50,100,200,400])
strength_star = strengths/(δ*ρi*g*H0)
fail_its_strength = [124,118,127,161]

plot(strength_star,fail_its_strength,'strength')

# %%



h = heights_norelax
t = fail_t_norelax

fig, ax = plt.subplots(1,1,figsize=(4,4))

ax.scatter(h, t, marker='o', label='Runs')

ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

coeffs = np.polyfit(np.log(h), np.log(t), 1)
x_fit = np.linspace(min(h)-20, max(h)+50, 100)
y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
ax.plot(x_fit, y_fit, ls='--', label=rf'Fit: $t \propto H^{{{coeffs[0]:.2f}}}$')

# x_fit = np.linspace(min(heights_divuvnew)-50, max(heights_divuvnew)+50, 100)

# just fit to y = Ax^n
# n = -3
# A = t[-2]/h[-2]**n
# y_fit = A * x_fit**n
# ax.plot(x_fit, y_fit, ls='--', label=rf'$t \propto H^{{{n}}}$')

ax.legend(ncol=2)#, bbox_to_anchor=(1, 1),ncol=2)

fig.savefig('iceberg_time_to_failure_vs_height.png', dpi=300, bbox_inches='tight')
fig.savefig('iceberg_time_to_failure_vs_height.pdf',bbox_inches='tight')


#%%




fig, ax = plt.subplots(figsize=(4, 4))

confidence_plot(ax,h_n3_dstar10, t_n3_dstar10, 3)


ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

ax.legend()

fig.savefig('iceberg_time_to_failure_vs_height.png',
            dpi=300, bbox_inches='tight')
fig.savefig('iceberg_time_to_failure_vs_height.pdf',
            bbox_inches='tight')
# %%
