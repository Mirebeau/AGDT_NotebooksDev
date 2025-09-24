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
#    print("np,μ,X",μ,X)
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
#    print(f"{Λ=}")
#    print(hessian)

    if μflt: 
        gradient = fltsym_iso(gradient)
        hessian = fltsym_iso(np.moveaxis(fltsym_iso(hessian),0,2))
#        print("hessian",hessian)
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
#            print("Newton py : ",μ,X,val,grad,desc)
            μ += desc
    
    # Extract the solution
    X[:n,:n]-= expsym_iso(μ) 
    λ,U = eigh(X)
    λm = np.maximum(0,-λ) 
    Δm = np.einsum('ik...,k...,jk...->ij...',U,λm,U) # (X-μ)_-
#    assert np.allclose(μ,Δm[0,0]) # Guaranteed from the optimality conditions
    return Δm[n:,n:], -Δm[n:,:n]*s2


def prox_perspective_matrix_rotated(τ,ρ,m,**kwargs):
    """
    This version is useful if m has size d,n with n>d. 
    (Reduces to size d,d using rotational invariance)
    """
    R,m1 = np.linalg.qr(m.T)
    ρ_,m_ = prox_perspective_matrix(τ,ρ,m1.T,**kwargs)
    return ρ_,m_@R.T





if False:
    np.random.seed(42)
    d = 2
    n = 2
    m_ = np.random.rand(d,n)-0.5
    ρ_ = pyRandomSym(d)

    # Slow to compile and some numerical error in dimension d+n=4, likely due to my sym_eig


    ρ1,m1 = prox_perspective_matrix(1.,ρ_,m_) #,niter=1)

    R = lp.rotation(0.3)

    ρ2,m2 = prox_perspective_matrix(1.,ρ_,m_@R) #,niter=1)

    print(np.max(np.abs(ρ1-ρ2)))
    print(np.max(np.abs(m1-m2@R.T))) # Numerical error on m is substantial...

    assert np.allclose(ρ1,ρ2)
    assert np.allclose(m1,m2@R.T,atol=1e-7)


np.random.seed(42)
d=2
n=3
m = np.random.rand(d,n)-0.5
ρ = pyRandomSym(d)


@ti.kernel
def test_qr():
    m0 = ti.Matrix(m)
    print(m0)
    if True:
        q,r = MP.qr(m0,True)
        print(q,r)
        print(m0-r@q)

    if True:
        q,r = MP.qr(m0)
        print(q,r)
        print(m0-q@r)

    if True:
        q,r = MP.qr(m0.transpose(),True)
        print(q,r)
        print(m0.transpose()-r@q)

    if True:
        q,r = MP.qr(m0.transpose())
        print(q,r)
        print(m0.transpose()-q@r)



if False: test_qr()


# Slow to compile and some numerical error in dimension d+n=4, likely due to my sym_eig


ρ1,m1 = prox_perspective_matrix(1.,ρ,m) #,niter=1)
ρ2,m2 = prox_perspective_matrix_rotated(1.,ρ,m) #,niter=1)

print(np.max(np.abs(ρ1-ρ2)))
print(np.max(np.abs(m1-m2))) 

@ti.kernel
def test_prox_perspective_qr():
    for _ in range(1):
        ρ_,m_ = ti.Matrix(ρ),ti.Matrix(m)
        ρ3,m3  = MP.prox_perspective_qr(ρ_,m_)
        print(ρ3-ti.Matrix(ρ1))
        print(m3-ti.Matrix(m1))

test_prox_perspective_qr()

exit(0)

R,m_ = np.linalg.qr(m.T); m_=m_.T; R=R.T
print(R,m_)
print(m_.shape,m.shape,R.shape)
print(m - m_@R)
print(R@R.T)

ρ2,m2 = prox_perspective_matrix(1.,ρ,m_) #,niter=1)

print(m1.shape,m2.shape,R.shape)

print(np.max(np.abs(ρ1-ρ2)))
print(np.max(np.abs(m1-m2@R))) 

exit(0)

assert np.allclose(ρ1,ρ2)
assert np.allclose(m1,m2@R,atol=1e-7)


# Now, take advantage of this invariance, in the case where m is a large vector 
#def prox_perspective_matrix_rotated(τ,ρ,m,**kwargs):





