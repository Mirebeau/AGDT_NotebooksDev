import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
import taichi as ti
from matplotlib import pyplot as plt

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t) # Note : debug=True causes segfault...
np.set_printoptions(linewidth=2000)

import agdt
from agdt.Waves.AnisoScalar import AnisoScalar,BoxNormal,Damping
np_float_t = agdt.convert_dtype['np'][float_t]

def ΔGauss(x): 
	"""Laplacian of Gaussian"""
	x2 = np.sum(x**2,axis=0)
	return (len(x)-x2)*np.exp(-x2/2)

Nx = 100
X,dx = np.linspace(-1,1,Nx,retstep=True)
X = X[None,:]
std = 3*dx
q_np = ΔGauss((X-dx/2)/std)
q_np = np.random.rand(Nx)
p_np = np.random.rand(Nx)
wavelength = std*4 # Typical wavelength of the signal

shape = (Nx,)
E = np.eye(1).astype(int)
decompdim = E.shape[1]
μ = np.ones(shape)
λ = np.ones((*shape,decompdim))

speed = 1
period = wavelength / speed # Period associated to the wavelength

normal = BoxNormal(shape,sides=((True,True),),width=2)
γ=None # Damping is not the same in the two formulations
γ = Damping(shape,sides=((True,True),),width=wavelength/dx,order=2.5) / period # 3* Damping coefficient
# Note : the optimal value is to divide by 2*period (with order=2), but this is likely due to the 
# unreasonable effectiveness of absorbing b.c in dimension one.
#print(normal[0],normal[-1])

dt = dx
wave = AnisoScalar(μ,λ,E,dx,dt,γ,normal=normal)
#print(wave.Γv)

q = wave.empty_like_v(); p = wave.empty_like_v();
p.from_numpy(p_np)
q.from_numpy(q_np)
v = wave.p2v(p)
σ = wave.q2σ(q)


if False:
	# ------- Testing the Verlet q -----
	# ?? Occasional segfault in the next code ?? (Without any out-of-bounds access detected by debug.)
	Href = wave.Hqp(q,p,'q')
	assert np.allclose(Href,wave.Hσv(σ,v,'σ'))
	#assert np.allclose(wave.Hqp(q,p,'q'),wave.Hσv(σ,v,'σ'))

	self=wave
	@ti.kernel
	def Verlet_σ(
		σ:ti.template(),  # field(float_t,(size,decompdim)) [INOUT]
		v:ti.template()): # field(float_t,size)             [INOUT]
		"""One Vertlet_v timestep (update v first) in the velocity-stress coordinates.
		Includes the damping of velocity and stress."""
		μ = ti.static(self.μ)
		for I in μ: self.update_σ(σ,v,I)
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: self.update_σ(σ,v,I)

	@ti.kernel
	def Verlet_q(
		q:ti.template(),  # field(float_t,(size,decompdim)) [INOUT]
		p:ti.template()): # field(float_t,size)             [INOUT]
		"""One Vertlet_v timestep (update v first) in the velocity-stress coordinates.
		Includes the damping of velocity and stress."""
		μ,τ = ti.static(self.μ,self.τ)
		for I in μ: q[I] += τ*μ[I]*p[I] # Update q (double timestep)
		for I in μ: self.update_p(q,p,I) # Update p
		for I in μ: self.update_p(q,p,I) # Update p, again
		for I in μ: q[I] += τ*μ[I]*p[I] # Update q (double timestep)
		
	Verlet_σ(σ,v)
	Verlet_q(q,p)
	if γ is None and np.allclose(normal,0): assert np.allclose(Href,wave.Hqp(q,p,'q'))
	else: Href = wave.Hqp(q,p,'q')
	assert np.allclose(Href,wave.Hσv(σ,v,'σ'))

if False:
	# ----------- Testing Verlet p -------
	Href = wave.Hqp(q,p)
	wave.Verlet_p(q,p)
	wave.Verlet_v(σ,v)

	assert np.allclose(σ.to_numpy(),wave.q2σ(q).to_numpy()) 
	assert np.allclose(v.to_numpy(),wave.p2v(p).to_numpy())
	if γ is None and np.allclose(normal,0): assert np.allclose(Href,wave.Hqp(q,p))
	else: Href = wave.Hqp(q,p)
	assert np.allclose(Href,wave.Hσv(σ,v))


if False:
	# --------------- Testing mixed formulation ------------
	qref = wave.empty_like_v(); pref = wave.empty_like_v(); σref = wave.empty_like_σ(); vref = wave.empty_like_v()
	qref.copy_from(q); pref.copy_from(p); σref.copy_from(σ); vref.copy_from(v)

	wave.mixed_setup()
	print("indσ, σind, ΓDσ",wave.indσ,wave.σind,wave.ΓDσ)
	Dσ = wave.mixed_empty_like_Dσ(σ)

	#print(σ,Dσ)
	#print(p.to_numpy())

	#print(q,σ,Dσ)


	for it in range(2):
		wave.mixed_Verlet_v(q,Dσ,v)
		wave.Verlet_v(σref,vref)
		#wave.Verlet_p(qref,pref) # Equivalent only if no damping

	#print(v.to_numpy()-vref.to_numpy())

	σmask = wave.mixed_σmask()
	qmask = np.logical_not(σmask)
	assert np.allclose(v.to_numpy(),vref.to_numpy())
	assert np.allclose(Dσ.to_numpy(),wave.mixed_empty_like_Dσ(σref).to_numpy())
	assert np.allclose(wave.qDσ2σ(q,Dσ).to_numpy(),σref.to_numpy())
	#assert np.allclose(v.to_numpy(),wave.p2v(pref).to_numpy())
	#assert np.allclose(q.to_numpy(),qref.to_numpy())
	#assert np.allclose(q.to_numpy()[qmask],qref.to_numpy()[qmask])

	#assert np.allclose(p.to_numpy(),pref.to_numpy())
	#γ_ = np.zeros(q_np.size)
	#wave.extended_setup(γ_)

if False:
	# ---- Testing extended formulation, constant damping ----
	γ_ = np.ones_like(γ) # We use a constant damping, in such way that the formulations should be equivalent
	wave = AnisoScalar(μ,λ,E,dx,dt,γ_,normal=normal)

	wave.extended_setup(γ_)
	divσ,σgradγ = wave.extended_divσ_σgradγ(σ)

	σref = wave.empty_like_σ(); vref = wave.empty_like_v(); σref.copy_from(σ); vref.copy_from(v)

	for it in range(3):
		wave.extended_Verlet(v,divσ,σgradγ)
		wave.Verlet_v(σref,vref)

	assert np.allclose(v.to_numpy(),vref.to_numpy())


if False:
	# ------ Testing extended formulation, non-constant damping ------
	# Seems to work well, numerical error is not discernible on this test example. 
	# Note that we use a smooth γ, supported in the domain interior

	def sSign(x): return np.where(np.abs(x)>1,np.sign(x),x*(15+x**2*(-10+x**2*3))/8)
	γ_ = 0.5*(1+sSign(3*(abs(X[0])-0.5)))
	#plt.plot(*X,γ_); plt.show()
	wave = AnisoScalar(μ,λ,E,dx,dt,γ_,normal=normal)
	q_np = ΔGauss((X-dx/2)/std)
	p_np = np.zeros_like(q_np)
	q = wave.empty_like_v(); p = wave.empty_like_v();
	p.from_numpy(p_np); q.from_numpy(q_np)

	σ,v = wave.q2σ(q),wave.p2v(p)
	σref = wave.empty_like_σ(); vref = wave.empty_like_v(); σref.copy_from(σ); vref.copy_from(v)

	wave.extended_setup(γ_)
	divσ,σgradγ=wave.extended_divσ_σgradγ(σ)

	for it in range(40):
		wave.extended_Verlet(v,divσ,σgradγ)
		wave.Verlet_v(σref,vref)

	plt.plot(*X,v.to_numpy(),label='v')
	plt.plot(*X,vref,label='vref')

	divσref,σgradγref=wave.extended_divσ_σgradγ(σref)
	plt.plot(*X,divσ,label='divσ')
	plt.plot(*X,divσref,label='divσref')

	plt.plot(*X,σgradγ.to_numpy(),label='σgradγ')
	plt.plot(*X,σgradγref,label='σgradγref')

	plt.legend()
	plt.show()

