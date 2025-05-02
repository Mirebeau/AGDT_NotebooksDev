# We check the implementation of two-dimensional absorbing boundary conditions and damping.

import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
import taichi as ti
from matplotlib import pyplot as plt

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)
np.set_printoptions(linewidth=2000)

import agdt
from agdt.Waves.AnisoScalar import AnisoScalar,BoxNormal,Damping
np_float_t = agdt.convert_dtype['np'][float_t]

def ΔGauss(x): 
	"""Laplacian of Gaussian"""
	x2 = np.sum(x**2,axis=0)
	return (len(x)-x2)*np.exp(-x2/2)

Nx = 100
aX,dx = np.linspace(-1,1,Nx,retstep=True)
X = np.meshgrid(aX,aX,indexing='ij')
v0_std = 3*dx
v0_np = ΔGauss((X-dx/2)/v0_std)
wavelength = v0_std*4 # Typical wavelength of the signal

shape = X[0].shape
vdim = X[0].ndim
E = np.eye(vdim).astype(int)
decompdim = E.shape[1]
μ = np.ones(shape)
λ = np.ones((*shape,decompdim))

speed = 1
period = wavelength / speed # Period associated to the wavelength

normal = BoxNormal(shape,sides=((True,True),(True,True)),width=2)
#γ = Damping(shape,((True,True),(True,True)),3*wavelength/dx,order=2.5) *0.25 / period # Damping coefficient
γ = Damping(shape,sides=((True,True),(True,True)),width=3*wavelength/dx,order=2) *0.4 / period # Damping coefficient


#print(normal[0],normal[-1])

#plt.contourf(γ);plt.colorbar();plt.show()

dt = dx/np.sqrt(vdim)
wave = AnisoScalar(μ,λ,E,dx,dt,γ,normal=normal)
#print(wave.Γv)
#print(wave.mE)

v = wave.empty_like_v(); v.from_numpy(v0_np.reshape(-1))
σ = wave.empty_like_σ(); σ.fill(0)
H_init = wave.Hσv(σ,v)

def show_contourf(v):
	plt.contourf(*X,v.to_numpy().reshape(shape))
	plt.axis('equal')
	plt.colorbar()
	plt.show()

Nt = 150 #150 #150
for it in range(Nt):
	wave.Verlet_v(σ,v)

H_final = wave.Hσv(σ,v)
print("Sqrt of energy decay ratio :",np.sqrt(H_init/H_final)) # Sqrt of energy is similar to amplitude

show_contourf(v)