"""
In this file, we implement the BBB formulation of EDOs, using taichi reverse autodiff.
EDO solved : v'+ L(v x v)/2 = 0, v(0)=v0
"""

agd_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations"; agdt_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi"
import sys; sys.path.insert(0,agd_path); sys.path.insert(0,agdt_path)

import taichi as ti
import numpy as np
from agd import LinearParallel as lp
from agd import AutomaticDifferentiation as ad

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t) # Default data types are f32 et i32
VectorType = ti.lang.matrix.VectorType

def L_Ricatti(α): return 2*α*np.ones( (1,1,1) )
def L_Euler(α): return np.array([α*np.array([[0,1],[1,0]]),np.zeros((2,2))])
def L_LotkaVolterra(α,β,γ,δ): return np.array([ [[0,β,-α],[β,0,0],[-α,0,0]], [[0,-δ,0],[-δ,0,γ],[0,γ,0]], np.zeros((3,3))])

def mk_BBB_EDO(L,v0,τ,γ,ε):
    d = len(L)
    assert L.shape == (d,d,d)
    assert v0.shape== (d,)

    L_np = L; L = ti.Matrix.field(d,d,float_t,shape=d); L.from_numpy(L_np)
    v0_np = v0; v0 = ti.Vector.field(d,float_t,shape=()); v0.from_numpy(v0_np) #VectorType(d,float_t); 

    @ti.kernel
    def obj_1(ϕ:ti.template(),res:ti.template()):
        N,ε2τ2 = ti.static(ϕ.shape[0],ε*2*τ**2)
        for n in ϕ:
            Lϕ = ti.math.exp(-γ*n*τ) * ti.math.eye(d)
            for i in ti.static(range(d)): Lϕ += ϕ[n][i]*L[i]
            iLϕ = ti.math.inverse(Lϕ)
            if n<N-1: Dtϕ = ϕ[n+1]-ϕ[n]; res[None] += Dtϕ @ iLϕ @ Dtϕ 
            if n>0:   Dtϕ = ϕ[n]-ϕ[n-1]; res[None] += Dtϕ @ iLϕ @ Dtϕ 
            res[None] -= ε2τ2*ti.log(ti.math.determinant(Lϕ))
    @ti.kernel # Autodiff requires splitting the kernels (for vs non-for operations)
    def obj_2(ϕ:ti.template(),res:ti.template()):
        res[None] = res[None]/(2*τ**2) + ϕ[0] @ v0[None] 

    def obj(ϕ,res): res[None]=0; obj_1(ϕ,res); obj_2(ϕ,res)

    def obj_np(ϕ):
        Lϕ = sum(ϕ[None,None,:,i]*L_np[i,:,:,None] for i in range(d))
        Γ = np.exp(-γ*τ*np.arange(len(ϕ)))
        for i in range(d): Lϕ[i,i] += Γ
        iLϕ = lp.inverse(Lϕ)
        Dtϕ = (ϕ[1:]-ϕ[:-1]).T
        iLϕ = iLϕ[:,:,1:]+iLϕ[:,:,:-1] # Harmonic mean
        return ( np.sum(lp.dot_VAV(Dtϕ,iLϕ,Dtϕ) )/(2*τ**2)  # Perspective function
            + lp.dot_VV(ϕ[0],v0_np) # Initial condition
            - ε*np.sum(np.log(lp.det(Lϕ))) # Positive definiteness enforcing penalty
            )

    # Evaluate both the objective function and its derivative
    ϕ_tiad = None
    res_tiad = ti.field(float_t,shape=tuple(),needs_grad=True)

    def Dobj(ϕ):
        nonlocal ϕ_tiad
        if ϕ_tiad is None: ϕ_tiad = ti.Vector.field(2,float_t,len(ϕ)+1,needs_grad=True)
        ϕ_tiad.from_numpy(np.concatenate((ϕ,np.zeros((1,d))),axis=0))
        with ti.ad.Tape(loss=res_tiad): obj(ϕ_tiad,res_tiad)
        return res_tiad[None],ϕ_tiad.grad.to_numpy()[:-1]

    ϕ_npad = None
    def Dobj_np(ϕ):
        nonlocal ϕ_npad
        if ϕ_npad is None: ϕ_npad = ad.Sparse.identity(shape = (ϕ.shape[0]+1,ϕ.shape[1]))
        ϕ_npad.value[:-1] = ϕ # Padding with zeros for the last coordinate
        obj_ad = obj_np(ϕ_npad)
        return obj_ad.value,obj_ad.to_dense().gradient()[:-2].reshape(ϕ.shape)

    return obj,obj_py,Dobj,Dobj_py

def mk_BBB_Burgers(v0,dt,dx,γ,ε):
    @ti.kernel
    def obj_ti(ϕ:ti.template(),res:ti.template()):
        T,X,idx,idx2 = ti.static(ϕ.shape(0),ϕ.shape[1],dx**-1,dx**-2)
        for t,x in ϕ:
            xm = (x+X-1)%X; xp = (x+1)%X # Periodicity in space
            m = (ϕ[t+1,x]-ϕ[t,x])/dt
            Dρ0 = (ϕ[t  ,xp]-2*ϕ[t  ,x]+ϕ[t  ,xm])*idx2
            Dρ1 = (ϕ[t+1,xp]-2*ϕ[t+1,x]+ϕ[t+1,xm])*idx2
            Γ0 = ti.math.exp(-γ*dt*t) 
            Γ1 = ti.math.exp(-γ*dt*(t+1)) 
            ρ0m = Γ0 - (ϕ[t  ,x ]-ϕ[t  ,xm])*idx
            ρ0p = Γ0 - (ϕ[t  ,xp]-ϕ[t  ,x ])*idx
            ρ1m = Γ1 - (ϕ[t+1,x ]-ϕ[t+1,xm])*idx
            ρ1p = Γ1 - (ϕ[t+1,xp]-ϕ[t+1,x ])*idx

            res[None] += 0.25*dt * ( 
                (m-ν*Dρ0)**2*(1./ρ0m+1./ρ0p) + (m-ν*Dρ1)**2*(1./ρ1m+1./ρ1p) 
                - ε*ti.math.log(ρ0m*ρ0p*ρ1m*ρ1p) )
            if t==0: res[None] += ϕ[t,x]*v0[x]

    def obj_np(ϕ):
        m = (ϕ[1:]-ϕ[:-1])/dt
        Dρ = (np.roll(ϕ,1,axis=1)-2*ϕ+np.roll(ϕ,-1,axis=1))/dx**2
        Γ = np.exp(-γ*dt*np.arange(len(ϕ)))
        ρ = Γ - (np.roll(ϕ,1,axis=1)-ϕ)/dx
        return 0.25*dt*(
            (m - ν*Dρ[:-1])**2 * (1/ρ[:-1]+1/np.roll(ρ[:-1],1,axis=1)) + 
            (m - ν*Dρ[1: ])**2 * (1/ρ[1: ]+1/np.roll(ρ[1: ],1,axis=1)) + 
            - 2*ε * np.sum(np.log(ρ[1:]*ρ[:-1]))
            ) + np.sum(ϕ[0]*v0)


    return obj_ti,obj_np

def mk_BBB_DF_ti(f,ϕVector=False):
    """Pad with zeros and compute the gradient via backprop"""
    ϕ_ad = None
    res_ad = ti.field(float_t,shape=tuple(),needs_grad=True)
    def Df(ϕ):
        nonlocal ϕ_ad
        T,X = ϕ.shape[0],ϕ.shape[1:]
        if ϕ_ad is None: 
            if ϕVector: ϕ_ad = ti.Vector.field(X[-1],float_t,(T+1,*X[:-1]),needs_grad=True)
            else: ϕ_ad = ti.field(float_t,(T+1,*X),needs_grad=True)
        ϕ_ad.from_numpy(np.concatenate((ϕ,np.zeros((1,*X))),axis=0))
        with ti.ad.Tape(loss=res_ad): f(ϕ_ad,res_ad)
        return res_ad[None],ϕ_ad.grad.to_numpy()[:-1]
    return Df

def mk_BBB_DF_np(f):
    """Pad with zeros and compute the gradient, via sparse forward differentiation"""
    ϕ_ad = None
    def Df(ϕ):
        nonlocal ϕ_ad
        T,X = ϕ.shape[0],ϕ.shape[1:]
        if ϕ_ad is None: ϕ_ad = ad.Sparse.identity(shape=(T+1,*X))
        ϕ_ad.value[:-1] = ϕ
        f_ad = f(ϕ_ad).to_dense()
        return f_ad.value,f_ad.gradient()[:ϕ.size].reshape(ϕ.shape)





T = 0.3
N = 5
L = L_Euler(1)
v0 = np.array([1,1.])
τ = T/(N-1)
γ = 0
ε = 1e-3

obj,obj_py,Dobj,Dobj_py = mk_BBB(L,v0,τ,γ,ε)

np.random.seed(42)
ϕ_np = np.random.rand(N+1,2); ϕ_np[-1]=0
ϕ = ti.Vector.field(2,float_t,N+1,needs_grad=True) 
ϕ.from_numpy(ϕ_np)
res = ti.field(float_t,shape=tuple(),needs_grad=True)
res[None]=0

obj(ϕ,res)
#obj_2(ϕ,res)
assert np.allclose(res[None],obj_py(ϕ_np)) # Check that obj is same

with ti.ad.Tape(loss=res): 
    obj(ϕ,res)
#    obj_2(ϕ,res)
    
ϕ_ad = ad.Dense.identity(constant=ϕ_np)
res_ad = obj_py(ϕ_ad)

assert np.allclose(ϕ.grad.to_numpy(), res_ad.gradient().reshape(ϕ_np.shape)) # Check that value is same
print(res[None],ϕ.grad.to_numpy())

val,grad = Dobj(ϕ_np[:-1])
val_ti,grad_ti = Dobj_py(ϕ_np[:-1])
assert np.allclose(val,val_ti)
assert np.allclose(grad,grad_ti)

def gradient_descent(Df,x,τ=1,nitermax=100,τmin=1e-6,grad_stopratio=1e-6):
    """Gradient descent with variable timestep"""
    val,grad = Df(x)
    gradnorm_orig = np.linalg.norm(grad)
    for niter in range(nitermax):
        gradnorm = np.linalg.norm(grad)
        if gradnorm<gradnorm_orig*grad_stopratio: return x,niter # Terminate if gradient is sufficiently reduced
        x_ = x - τ*grad
        val_,grad_ = Df(x_)
        df_lin = τ*gradnorm**2 # expected improvement, if f was linear
        df = val-val_ # Measured improvement
        if τ>τmin and not (df >= 0.2*df_lin): τ=max(τ/2,τmin); continue # Too non-linear behavior
        if df>0.8*df_lin: τ*=1.5; # Linear behavior, increase step
        x = x_; val = val_; grad = grad_ 
        #print(f"{τ=}, {x=}, {val=}")
        print(f"{τ=}")
    print(f"Reached {nitermax=} with gradient norm reduction {gradnorm/gradnorm_orig=}")
    return x,nitermax

def ftest(x): return x[0]**2+10*x[1]**2
def mk_Df(f,x0):
    x_ad = ad.Dense.identity(constant=x0)
    def Df(x):
        x_ad.value=x
        f_ad = f(x_ad)
        return f_ad.value,f_ad.gradient()
    return Df

if False:
    x0 = np.array([1.,1.])
    x_opt,niter = gradient_descent(mk_Df(ftest,x0),x0)
    print(x_opt)
    assert np.allclose(x_opt,0,atol=1e-5)

ϕ0 = np.zeros((N,2))
ϕ_opt,niter = gradient_descent(Dobj,ϕ0,nitermax=500)

print(f"{niter=}")
print(Dobj_py(ϕ_opt))


# Now, I want to do some optimization. Can use lbfgs, or gradient descent.

#print(ϕ.grad,res_ad.gradient().reshape(ϕ_np.shape))


#print(f"{res[None]=}")
#print(obj_py(ϕ_np))

