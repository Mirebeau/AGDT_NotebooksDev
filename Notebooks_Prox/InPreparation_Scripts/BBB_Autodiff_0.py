agd_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations"; agdt_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi"
import sys; sys.path.insert(0,agd_path); sys.path.insert(0,agdt_path)

import taichi as ti
import numpy as np
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t) # Default data types are f32 et i32


n=5
d=2
vec_t = ti.lang.matrix.VectorType(d,float_t)
mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)

ϕ = ti.Vector.field(d,float_t,shape=n,needs_grad=True)
obj = ti.field(float_t,shape=(),needs_grad=True)

ϕ.from_numpy(np.arange(n*d).reshape((n,d)))
print(ϕ)


@ti.kernel
def BBB0():
	obj[None] = ϕ[0][0]+ϕ[1][1] # Needs obj[None]

@ti.kernel
def BBB1():
	obj[None] = ϕ[1][0]-ϕ[0][1] # Needs obj[None]

@ti.kernel
def BBB2(ϕ_:ti.template(),obj_:ti.template()):
	obj_[None] = ϕ[1][1]*ϕ[2][0]

with ti.ad.Tape(loss=obj):
	BBB0()
print(obj)
print(ϕ.grad)

with ti.ad.Tape(loss=obj): # Repetition ok
	BBB1()
print(obj)
print(ϕ.grad)

with ti.ad.Tape(loss=obj): # Parameters ok
	BBB2(ϕ,obj)
print(obj)
print(ϕ.grad)

@ti.kernel
def BBB3(ϕ:ti.template(),
	τ:float_t,
	L:ti.template(),
	v0:ti.template(),
	ε:float_t,
	result:ti.template()):
	for i in ϕ:
		result[None]+=ϕ[i][i%2]

τ = 1 #.float_t(1.)
L = ti.Matrix.field(d,d,float_t,shape=d)
L.from_numpy(np.arange(d*d*d).reshape(d,d,d))
v0 = ti.math.vec2((1,2.))
ε = 1. #float_t(1.)

with ti.ad.Tape(loss=obj):
	BBB3(ϕ,τ,L,v0,ε,obj)
print(obj)
print(ϕ.grad)

nT = ϕ.shape[0]-1
γ = 1.
@ti.kernel
def BBB4(ϕ:ti.template(),
	τ:float_t,
	L:ti.template(),
	v0:ti.template(),
	ε:float_t,
	result:ti.template()):
	nT,γ = ti.static(ϕ.shape[0]-1,1.)
#	nT = ϕ.shape[0]-1
#	γ = 1.
	for i in ti.ndrange(nT):
		Dtϕ = (ϕ[i+1]-ϕ[i])/τ
		Γ = ti.math.exp(-γ*i*τ)
#		Lϕ = ϕ[i][0]*L[0] + ϕ[i][1]*L[1]
		Lϕ = Γ*ti.math.eye(d)
		for j in ti.static(range(d)):
			Lϕ += ϕ[i][j]*L[j]
#		iLϕ = Lϕ+1
		iLϕ = ti.math.inverse(Lϕ) # Need some time shift
		#iLϕ = mat_t(Lϕ[0,0],-Lϕ[0,1],-Lϕ[1,0],Lϕ[1,1])
		result[None] += Γ * (Dtϕ @ iLϕ @ Dtϕ)

with ti.ad.Tape(loss=obj):
	BBB4(ϕ,τ,L,v0,ε,obj)
print(obj)
print(ϕ.grad)

def BBB4_py(ϕ,τ,L,v0,ε):
	"""Python implementation, with basic forward AD, just to be sure."""
	Dtϕ = (ϕ[1:]-ϕ[:-1]).T/τ
	Γ = np.exp(-γ*τ*np.arange(nT))
	Lϕ = sum(ϕ[None,None,:-1,i]*L[i,:,:,None] for i in range(d))
	for i in range(d): Lϕ[i,i] += Γ
	iLϕ = lp.inverse(Lϕ)
	return np.sum(Γ * lp.dot_VAV(Dtϕ,iLϕ,Dtϕ))


ϕ_ad = ad.Dense.identity(constant=ϕ.to_numpy())
res = BBB4_py(ϕ_ad,τ,L.to_numpy(),v0.to_numpy(),ε)


print(res)

# @ti.kernel
# def obj(ϕ:ti.types.ndarray(ndim=2),
# 	τ,L,v0,ε):
# 	"""
# 	Fonction objectif the la formulation BBB de l'EDO: v'+ L(v x v)/2 = 0, v(0)=v0
#     ϕ (tableau, de taille (nT,d))
#     τ (réel) pas de temps
#     L (tableau, de taille (d,d,d))
#     v0 (tableau, de taille (d))
#     ϵ (réel>0) : paramètre de relaxation pour la contrainte Id + L^*(ϕ) >= 0.
#     """

# Ricatti equation (just to test)



# Solve the exponential, with a weight