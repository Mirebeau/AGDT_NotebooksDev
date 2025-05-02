# We test the implementation of the AnisoWave class for the linear anisotropic wave eqn,
# with some actual anisotropy


import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
from agdt import Selling

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)
np.set_printoptions(linewidth=2000)
π = np.pi

import agdt
from agdt.Waves.AnisoScalar import AnisoScalar,BoxNormal,Damping
np_float_t = agdt.convert_dtype['np'][float_t]

def ΔGauss(x): 
	"""Laplacian of Gaussian, used as source."""
	x2 = np.sum(x**2,axis=0)
	return (len(x)-x2)*np.exp(-x2/2)

Nx = 100
aX,dx = np.linspace(-1,1,Nx,retstep=True)
X = np.array(np.meshgrid(aX,aX,indexing='ij'))
v0_std = 3*dx
v0_np = ΔGauss((X-dx/2)/v0_std)
wavelength = v0_std*4 # Typical wavelength of the signal

shape = X[0].shape
vdim = X[0].ndim

# ------- Build some anisotropic tensor field ------

D = np.zeros((*shape,vdim,vdim))
mat_t = ti.lang.matrix.MatrixType(vdim,vdim,2,float_t)
Diag = mat_t( (2.**2,0),(0,1.**2) )
decompdim = 3
λ = np.zeros((*shape,decompdim))
e = np.zeros((*shape,decompdim,vdim),dtype=np.int8)
μ = np.ones(shape)

@ti.kernel
def set_D(D : ti.types.ndarray(dtype=mat_t,ndim=vdim),
	λ : ti.types.ndarray(dtype=Selling.weights_t(vdim),ndim=vdim),
	e : ti.types.ndarray(dtype=Selling.offsets_t(vdim),ndim=vdim)):
	for i,j in D:
		θ = ti.math.cos(0.7 * 2*π * i*dx)
		R = ti.math.rotation2d(θ)
		D[i,j] = R @ Diag @ R.transpose()
		λ[i,j],e[i,j] = Selling.decomp(D[i,j])
set_D(D,λ,e)

print(λ.shape,e.shape)
#print(Diag,D[10,10],λ[10,10])
λ,E = Selling.DecompWithFixedOffsets(λ,e)
print(λ.shape,E.shape)
print(E)
print(np.max(λ),np.min(λ))

E = np.eye(2).astype(int); λ = np.ones((*shape,2))

# Run the PDE, first without damping and absorbing b.c., to check chg of vars and energy conservation
wave = AnisoScalar(μ,λ,E,dx)
q = wave.empty_like_v()
p = wave.empty_like_v()
p.from_numpy(ΔGauss(X).reshape(-1))

σ = wave.q2σ(q)
v = wave.p2v(p)

Hinit = wave.Hqp(q,p)

for it in range(10):
	wave.Verlet_p(q,p)
	wave.Verlet_v(σ,v)

print(f"Conservation : {Hinit=}, {wave.Hqp(q,p)=}, {wave.Hσv(σ,v)=}")

print(f"Orig :  {wave.Hqp(q,p,'orig')=}, {wave.Hσv(σ,v,'orig')=}")
print(f"q :  {wave.Hqp(q,p,'q')=}, {wave.Hσv(σ,v,'σ')=}")

#print(f"Orig-p :  {wave.Hqp(q,p,'orig')-wave.Hqp(q,p,'p')=}, {wave.Hσv(σ,v,'orig')-wave.Hσv(σ,v,'v')=}")
#print(f"Orig-q :  {wave.Hqp(q,p,'orig')-wave.Hqp(q,p,'q')=}, {wave.Hσv(σ,v,'orig')-wave.Hσv(σ,v,'σ')=}")



print(np.max(np.abs(wave.p2v(p).to_numpy()-v.to_numpy())))
print(np.max(np.abs(wave.q2σ(q).to_numpy()-σ.to_numpy())))





#normal = BoxNormal(shape)
#damping = Damping(shape,width=)


