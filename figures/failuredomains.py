#%%
import numpy as np
import matplotlib.pyplot as plt

ν = 0.325
λ = 2*ν/(1-2*ν)
E = 2*(1+ν)
μ = 1.0




def free_energy_lo(εs):
    εs = np.sort(εs)
    ε1,ε2,ε3 = εs
    # ε3>ε2>ε1
    trε = ε1 + ε2 + ε3
    ψ1 = 0.5*λ/μ*trε**2 + (ε1**2 + ε2**2 + ε3**2)
    ψ2 = ψ1 - 0.5*E/μ*ε1**2
    ψ3 = (1+ν)*((1-ν)*ε3 + ν*ε2 + ν*ε1)**2/((1-2*ν)*(1-ν**2))
    
    if ε1>0:
        ψ = ψ1
    elif (ε2+ν*ε1)>0: 
        ψ = ψ2
    elif ((1-ν)*ε3 + ν*ε2 + ν*ε1)>0:
        ψ = ψ3
    else:
        ψ = 0
    return ψ


def free_energy_dp(ε,γ=np.sqrt(2)):
    K = λ/μ + 2/3
    trε = sum(ε)
    e = np.array([[ε[0], 0, 0], [0, ε[1], 0], [0, 0, ε[2]]])
    eD = e - 1/3*trε*np.eye(3)
    eD2 = np.einsum('ij,ji->',eD,eD) + 1e-16

    ψ1 = 0.5*K*trε**2 + eD2

    ψ2 = (K*γ*trε + 2*np.sqrt(eD2))**2/ (2*(K*γ**2 + 2))


    if np.sqrt(eD2)<trε/γ:
        return ψ1
    elif np.sqrt(eD2)< - γ*K/2*trε:
        return 0
    else:
        return ψ2
    


def stress_plus_dp(ε, γ=np.sqrt(2)):
    K = λ/μ + 2/3
    trε = sum(ε)
    e = np.array([[ε[0], 0, 0], [0, ε[1], 0], [0, 0, ε[2]]])
    eD = e - 1/3*trε*np.eye(3)
    eD2 = np.einsum('ij,ji->',eD,eD) + 1e-16
    δ = np.eye(3)
    

    σ1 = K*trε*δ + 2*eD
    σ2 = (K*γ*δ + eD/np.sqrt(eD2))/(K*γ**2 + 2)

    if np.sqrt(eD2)<trε/γ:
        return σ1
    elif np.sqrt(eD2)< - γ*K/2*trε:
        return 0*δ
    else:
        return σ2






def stress0(ε):
    return [λ*sum(ε)+μ*ε[0],2*λ*sum(ε)+2*μ*ε[1],λ*sum(ε)+2*μ*ε[2]]



# e1,e2 = np.meshgrid(np.linspace(-10,10,200),np.linspace(-10,10,200))
# e3 = 0.5*(e1+e2)
e1,e2,e3 = np.meshgrid(np.linspace(-10,10,50),np.linspace(-10,10,50),np.linspace(-10,10,50))
# e3 = np.zeros_like(e1)

ψ = 0.5*λ*(e1+e2+e3)**2 + μ*(e1**2+e2**2+e3**2)
γ = np.sqrt(2)
ψ = np.zeros_like(e1)
σ = np.zeros((e1.shape[0],e1.shape[1],e1.shape[2],3,3))
for i in range(e1.shape[0]):
    for j in range(e1.shape[1]):
        for k in range(e1.shape[2]):
            ψ[i,j,k] = free_energy_dp([e1[i,j,k],e2[i,j,k],e3[i,j,k]],γ)
            σ[i,j,k,...] = stress_plus_dp([e1[i,j,k],e2[i,j,k],e3[i,j,k]],γ)


fig, ax = plt.subplots()
# ax.contour(e1,e2,ψ)
ax.mesh(σ[...,0,0],σ[...,1,1],ψ,levels=[5])
ax.set_aspect('equal')
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.grid()
