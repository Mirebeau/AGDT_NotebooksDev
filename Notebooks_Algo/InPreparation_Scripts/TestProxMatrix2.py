import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")


import numpy as np
import taichi as ti
ti.init(arch=ti.cpu,default_fp=ti.f64)
from agdt.Proximal import MatrixPerspective as MP
from agdt import Linalg

from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
np.set_printoptions(linewidth=2000)


def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M

def _sym_iso(i,j): 
    """Factors used for isometry between Frobenius norm and Euclidean norm."""
    return np.sqrt(2) if i!=j else 1

def fltsym_iso(m):
    """Turns a symmetric matrix into a vector, isometrically w.r.t
    the Frobenius norm and the Euclidean norm."""
    return ad.array([m[i,j]*_sym_iso(i,j) for i in range(len(m)) for j in range(i+1)])

def expsym_iso(v):
    """Turns a vector into a symmetric matrix, isometrically w.r.t 
    the Frobenius norm and the Euclidean norm."""
    d = int(np.sqrt(2*len(v)))
    def index(i,j): return (max(i,j)*(max(i,j)+1))//2+min(i,j)      
    return ad.array([[v[index(i,j)]/_sym_iso(i,j) for i in range(d)] for j in range(d)])


def eigh(m):
    """np.linalg.eigh, with geometry first"""
    λ,U = np.linalg.eigh(np.moveaxis(m,(0,1),(-2,-1)))
    return np.moveaxis(λ,-1,0), np.moveaxis(U,(-2,-1),(0,1))

def g(μ,X,ret='value_gradient_hessian'):
    """
    Evaluate the function Tr(μ) + | (X-μ)_+ |_Fr^2, 
    as well as its gradient and Hessian. If μ has a smaller 
    size than X, then it is subtracted from the top-left corner.
    - μ : symmetric matrix
    - X : symmetric matrix
    - ret (string, optional) : 'value', 'value_gradient_hessian', 'value_gradient_direction'
    If μ.ndim == X.ndim-1, then μ is assumed to be in flattened form
    """
    print("np,μ,X",μ,X)
    μflt = μ.ndim==X.ndim-1
    if μflt: μ = expsym_iso(μ)
    
    # Construct Δ = X-μ
    n=len(μ)
    Δ = X.copy()
    Δ[:n,:n]-=μ 
    
    # Extract the positive part
    λ,U = eigh(Δ)
    λp = np.maximum(0,λ) 
    value = np.einsum('ii...',μ) + np.sum(λp**2)/2 # Value of the functional g

    if ret=='value': return value
        
    Δp = np.einsum('ik...,k...,jk...->ij...',U,λp,U) # (X-μ)_+
    gradient = np.eye(n).reshape((n,n)+(1,)*(X.ndim-2)) - Δp[:n,:n]
#    print("gradient",gradient)
    num = λp[None,:]+λp[:,None]
    den = np.abs(λ[None,:])+np.abs(λ[:,None])
    den[den==0]=1
    Λ = num/den
    V = U[:n]
    hessian = np.einsum("ij...,ki...,lj...,mi...,nj...->klmn...",Λ,V,V,V,V)
    print(f"{Λ=}")
#    print(hessian)

    if μflt: 
        gradient = fltsym_iso(gradient)
        hessian = fltsym_iso(np.moveaxis(fltsym_iso(hessian),0,2))
        print("hessian",hessian)
        if ret=='value_gradient_direction': return value, gradient, -lp.solve_AV(hessian,gradient)
    assert ret=='value_gradient_hessian'
    return value,gradient,hessian

def prox_perspective_matrix(τ,ρ,m,niter=None):
    """
    Proximal operator of the perspective function.
    - τ (real) : proximal time step
    - ρ (array) : symmetric array of shape (d,d,*s)
    - m (array) : array of shape (d,n,*s)
    - niter (optional, int) : if specified, a fixed number of basic Newton iterations are applied
    (Otherwise, using a damped Newton method with automatic stopping criterion.)
    """
    if τ!=1: 
        ρ1,m1 = prox_perspective_matrix(1,ρ/τ,m/τ,niter)
        return τ*ρ1,τ*m1
    # TODO : we could take advantage of some rotation invariance in the case n>d.
    # Build the block matrix
    d,n = m.shape[:2]
    shape = m.shape[2:]
    cat = np.concatenate
    eye = np.broadcast_to(np.eye(n,n).reshape( (n,n)+(1,)*(ρ.ndim-2)), (n,n)+shape)
    s2 = np.sqrt(2)
    X = cat( (cat((eye,np.moveaxis(m/s2,0,1)),axis=1), # (n,n),(n,d)
              cat((m/s2,-ρ),axis=1)), axis=0) # (d,n), (d,d)   
#    print("X np : ",X)

    # Solve the dual problem for μ using a Newton method
    μ = np.zeros(((n*(n+1))//2, *shape))
    if niter is None:
        μ = ad.Optimization.newton_minimize(lambda μ : g(μ,X,'value'), μ, 
        f_value_gradient_direction = lambda μ : g(μ,X,'value_gradient_direction'))
    else:
        for iter in range(niter):
            val,grad,desc = g(μ,X,'value_gradient_direction')
            print("Newton py : ",μ,X,val,grad,desc)
            μ += desc
    
    # Extract the solution
    X[:n,:n]-= expsym_iso(μ) 
    λ,U = eigh(X)
    λm = np.maximum(0,-λ) 
    Δm = np.einsum('ik...,k...,jk...->ij...',U,λm,U) # (X-μ)_-
#    assert np.allclose(μ,Δm[0,0]) # Guaranteed from the optimality conditions
    return Δm[n:,n:], -Δm[n:,:n]*s2



@ti.kernel
def test_sym2flt():
    for _ in range(1):
        r=0
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                k, = ti.static((Linalg.sym2flt_index(i,j),))
                r+=k
        print(r)

        r=0
        for k in range(10):
            i,j = ti.static(Linalg.flt2sym_index(k))
            r+=i+j
        print(r)
test_sym2flt()



if False:
    np.random.seed(42)
    X_ = pyRandomSym(3)
    μ_ = pyRandomSym(2)
#    μ_np = fltsym_iso(μ_)
#    μ_ = np.array([0.5,-0.2,0.7])
#    μ_ = np.array([1.])
    
    val,grad,direc = g(fltsym_iso(μ_),X_,'value_gradient_direction')
    print("val, grad, dir",val,grad,-expsym_iso(direc))
#    print("np value, gradient, hessian",g(fltsym_iso(μ_),X_))

    X = ti.Matrix(X_)
    μ = ti.Matrix(μ_)

    @ti.kernel
    def testPersp():
        for _ in range(1):
            val,grad,desc = MP._prox_perspective_dual_obj(Linalg.sym2flt(μ),X,True)
            #print("hess",hess)
            print("val, grad, dir",val,grad, desc)
            #print(val)
    testPersp()

#exit(0)

np.random.seed(42)
d=2
n=2
m_ = np.random.rand(d,n)-0.5
ρ_ = pyRandomSym(d)

# Slow to compile and some numerical error in dimension d+n=4, likely due to my sym_eig


print("Matrix prox np : ",prox_perspective_matrix(1.,ρ_,m_,niter=1))

m = ti.Matrix(m_)
ρ = ti.Matrix(ρ_)

#print(m)

@ti.kernel
def testProx():
    for _ in range(1):
        print("Matrix prox ti : ",MP._prox_perspective_Newton(ρ,m,1))
testProx()


