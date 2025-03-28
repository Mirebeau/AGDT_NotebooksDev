agd_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations"; agdt_path = "/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi"
import sys; sys.path.insert(0,agd_path); sys.path.insert(0,agdt_path)


# Imports from the AGD library
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
norm = ad.Optimization.norm
norminf = ad.Optimization.norm_infinity
from agd.Plotting import savefig,Tissot
def mixednorm(arr,p,q): return norm(norm(arr,q,axis=1,averaged=True),p,averaged=True)
from agd import Selling,Metrics
from agd.ExportedCode.Notebooks_Algo import TensorSelling as ts

# Imports from the AGDT library
from agdt.Proximal import Proj,QPME,Prox,Misc
from agdt.Proximal.Misc import convert_dtype,tifunc

# Import from standard Python libraries
import taichi as ti
import numpy as np
import scipy
from matplotlib import pyplot as plt
import copy
from scipy.sparse.linalg import lsqr as sparse_lsqr

cmul = ti.math.cmul; cconj = ti.math.cconj
π = np.pi
np.set_printoptions(linewidth=2000)
ti.init(arch=ti.cpu,default_fp=ti.f64) # Default data types are f32 et i32



device = 'mps'; # Appropriate for Apple M1
#device = 'cuda'; # Appropriate for Nvidia graphics cards

def CP_Aniso(u0,dt,dx,nT,λ,E,τ_f=1,verb=0,niter=1000,rhs=None):
    """
    Chambolle-Pock optimization method applied to the BBB formulation of the aniisotropic QPME, Dtu = (1/2) div(D grad u^2)
    Same as CP_Iso, but with offsets E and corresponding coefficients λ
    """
    # Normalization of the initial condition
    xp = Misc.get_array_module(u0)
    if xp is not np: u0_xp=u0; u0=u0.cpu().numpy() # Only for some cheap preprocessing...
    np_float_t = Misc.convert_dtype['np'][u0.dtype]
    Nu0 = np.sqrt(np.sum(u0**2)/np_float_t(u0.size)) # Averaged L2 norm of u0
    u0 = u0/Nu0
    dt = dt*Nu0

    # Initialization
    ρ = np.ones_like(u0,shape=(2*nT,*u0.shape))
    m = np.asarray([*(u0,)*nT,*(-u0,)*nT])
    if rhs is None: rhs = np.zeros_like(ρ[:nT+1])
    Rhs = Prox.mk_rhs(dt,u0,rhs)
    Ne = len(E)
    me = np.zeros_like(u0,shape=ρ.shape+(Ne,))
    μe = me.copy()
    if xp is not np: ρ,m,Rhs,me,μe = [Misc.asarray(e,like=u0_xp) for e in (ρ,m,Rhs,me,μe)]

    x = Misc.asobjarray(m,ρ,me,μe)
    y = 0*x; tx = copy.deepcopy(x) # Dual variable, additional variable
    dt,dx,τ_f = map(np_float_t,(dt,dx,τ_f))
    τ_gs = 1/τ_f # One needs τ_f τ_gs |K|^2 <1, but the coupling operator is the identity
    print(x[0].dtype,x[1].dtype,x[2].dtype,x[3].dtype,y[0].dtype,tx[0].dtype,type(τ_f),type(τ_gs))

    
    # Note : prox_f and proj_g are inplace operators by default (modifiy input)
    prox_f_obj = QPME.mk_prox_Iso(dt,dx,Rhs)
    proj_f_λ = QPME.mk_proj_λ(λ)
    @Misc.useobjarray
    def prox_f(m,ρ,me,μe,τ,inplace=True):
        return *prox_f_obj(m,ρ,τ,inplace),*proj_f_λ(me,μe,inplace)

    proj_g = Misc.useobjarray(QPME.mk_proj_E(dt,dx,E,ρ))        
    def prox_gs(y,τ=1): return y-proj_g(y,inplace=False) # Projection onto orthogonal space
    x,y,tx = Misc.ChambollePock_raw(x,y,tx,prox_f,prox_gs,τ_f,τ_gs,niter)
    return x[0]*Nu0,x[1],x[2]*Nu0,x[3]*Nu0  # Put back the normalization


def diffeo(X,ϵ=0.05): 
    """A measure preserving perturbation of the identity map on the torus (R/Z)^2, always invertible."""
    x,y = X
    x = x + ϵ*np.sin(2*π*y+1) 
    y = y + ϵ*np.sin(4*π*x+5) 
    x = x + ϵ*np.sin(4*π*y+3)
    y = y + ϵ*np.sin(2*π*x+2) 
    return ad.array([x,y])

def diffeo_lin(X,ϵ=0.05):
    x,y = X
    return ad.array([2*x,y])

    
def make_domain(Nx=50): 
    aX,dx = np.linspace(0,1,Nx,endpoint=False,retstep=True)
    aX+=dx/2
    X = np.array(np.meshgrid(aX,aX,indexing='ij'))
    return X,dx


nX = 256 if device=='cuda' else 50


diff = lambda X:diffeo(X-0.5,ϵ=0.035)
X,dx = make_domain(nX) 
X_ad = ad.Dense.identity(constant=X,shape_free=(2,))
dϕ = diff(X_ad).gradient()
D = Metrics.Riemann(np.eye(2)).transform(dϕ).m
#λ,e = ts.dense_decomp(*ts.smooth_decomp(D))
λ,e = ts.dense_decomp(*Selling.Decomposition(D))
assert np.allclose(np.sum(λ*lp.outer_self(e)[...,None,None],axis=2),D)
Λ = np.array([fd.AlignedSum(λi,e[:,i],(0,1),(1/2.,1/2.),padding=None) for i,λi in enumerate(λ)])
λ_ = np.ascontiguousarray(np.moveaxis(Λ,0,-1))

nT = 64
Ti = 2e-5
Tf = 1e-4; T = Tf-Ti
dt = T/nT

Y = diff(X) # Transformed coordinates
u0 = QPME.Barenblatt(Ti,Y)


#m_Aniso,ρ_Aniso,me_Aniso,μe_Aniso = CP_Aniso(u0,dt,dx,nT,λ_,e.T,τ_f=1,niter=5)


import torch
ti.init(arch=ti.gpu,default_fp=ti.f32)
u0_gpu = torch.asarray(u0.astype(np.float32),device=device)
λ_gpu = torch.asarray(λ_.astype(np.float32),device=device)

#torch.fft.fftn(λ_gpu)
m_Aniso,ρ_Aniso,me_Aniso,μe_Aniso = CP_Aniso(u0_gpu,dt,dx,nT,λ_gpu,e.T,τ_f=1,niter=50)
