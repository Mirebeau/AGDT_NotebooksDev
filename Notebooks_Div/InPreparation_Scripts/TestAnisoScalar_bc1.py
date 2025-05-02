# We check the implementation of one-dimensional absorbing boundary conditions and damping.
# Note that, in theory, b.c. should be perfectly absorbing (specific to the one-dimensional case)

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

def d2Gauss(x): 
	"""Second derivative of Gaussian"""
	return (1-x**2)*np.exp(-x**2/2)

if False:
	# Check the construction of the normal vector
	print("One dimensional normal vector")
	print(f"{BoxNormal( (15,), sides=((True,False),), width=7)=}")
	normal = BoxNormal( (7,8), sides=((True,True),(True,False)), width=4)
	print("Two dimensional normal vector")
	print(normal[...,0])
	print(normal[...,1])
	print(np.sum(normal**2,axis=-1))


if False:
	# Check the construction of the damping coefficient
	print(Damping((15,),sides=((True,False),), width=6))
	print(Damping((7,8),sides=((True,True),(True,False)), width=2))

Nx = 500
X,dx = np.linspace(-1,1,Nx,retstep=True)
σ0_std = 6*dx
σ0_np = d2Gauss((X-dx/2)/σ0_std)[:,None]
wavelength = σ0_std*4 # Typical wavelength of the signal

shape = (Nx,)
E = np.eye(1).astype(int)
decompdim = E.shape[1]
μ = np.ones(shape)
λ = np.ones((*shape,decompdim))

speed = 1
period = wavelength / speed # Period associated to the wavelength

normal = BoxNormal(shape,sides=((True,True),),width=2)
γ = Damping(shape,sides=((True,True),),width=3*wavelength/dx,order=2.5) / period # Damping coefficient
# Note : the optimal value is to divide by 2*period (with order=2), but this is likely due to the 
# unreasonable effectiveness of absorbing b.c in dimension one.
print(normal[0],normal[-1])

dt = dx
wave = AnisoScalar(μ,λ,E,dt,dx,γ,normal=normal)
print(wave.Γv)
#print(wave.mE)

σ = wave.empty_like_σ(); σ.from_numpy(σ0_np)
v = wave.empty_like_v(); v.fill(0)
H_init = wave.Hσv(σ,v)

Nt = 400 #150
for it in range(Nt):
	wave.Verlet_v(σ,v)

H_final = wave.Hσv(σ,v)
print("Sqrt of energy decay ratio :",np.sqrt(H_init/H_final)) # Sqrt of energy is similar to amplitude

plt.plot(X+dx/2,σ)
plt.plot(X,v)

plt.show()

#print(normal)
