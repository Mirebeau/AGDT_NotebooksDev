import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

import agdt
from agdt import Selling
import numpy as np
import taichi as ti
float_t = ti.f32
np_float_t = agdt.convert_dtype['np'][float_t]
ti.init(arch=ti.cpu,default_fp=float_t)

@ti.kernel # Function must be called within a kernel
def test_RandomSym():
    for _ in range(1): # Most functions need to be called within an inner loop
        m = Selling.random_sym(3,0.1)
        print(m)
test_RandomSym()


def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M

np.random.seed(42) # Reproducibility
for d in (2,3):
    # Define a symmetric matrix type
    mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)
    m = ti.field(mat_t,4)
    m.from_numpy(pyRandomSym(m.n,shape=m.shape).astype(np.float32))

    # Build the Selling decomposition routine
    λ = ti.field(Selling.weights_t(d),m.shape) # Weights
    e = ti.field(Selling.offsets_t(d),m.shape) # Offsets
    
    @ti.kernel
    def test_Selling():
        for I in m:
            λ[I],e[I] = Selling.decomp(m[I])
    test_Selling()
    print("Weights and offsets associated to the first matrix of the field")
    print(λ[0])
    print(e[0])

    m_rec = ti.field(mat_t,m.shape)
    @ti.kernel
    def test_Reconstruct():
        for I in m_rec:
            m_rec[I] = Selling.reconstruct(λ[I],e[I])
    test_Reconstruct()
    assert np.allclose(m_rec.to_numpy(),m.to_numpy())
    print("Reconstruction test passed")

    Λ_,E_ = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
    print("Weights associated to the first matrix of the field, common offsets")
    print(Λ_[0,:])
    print(E_)
    nE = len(E_)
    Λ = ti.field(ti.lang.matrix.VectorType(nE,float_t),m.shape); Λ.from_numpy(Λ_)
    E = ti.field(ti.lang.matrix.MatrixType(nE,m.n,2,e.dtype),shape=tuple()); E.from_numpy(E_)
    m_rec.fill(0)

    @ti.kernel
    def test_FixedOffsets():
        for I in m_rec:
            m_rec[I] = Selling.reconstruct(Λ[I],E[None])
    test_FixedOffsets()
    assert np.allclose(m_rec.to_numpy(),m.to_numpy())
    print("Fixed offsets test passed")
    print("-------------------------\n")



np.random.seed(36)
d = 2
mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)
nT = 50
T = ti.field(float_t,nT); T.from_numpy(np.linspace(0,1,nT).astype(np_float_t))
m = ti.field(mat_t,2); m.from_numpy(pyRandomSym(mat_t.n,shape=(2,)))
λ  = ti.field(Selling.weights_t(d),T.shape)
e  = ti.field(Selling.offsets_t(d),T.shape)
sdecompdim = Selling.decompdim(d,smooth=True)
sλ = ti.field(Selling.weights_t(d,sdecompdim),T.shape)
se = ti.field(Selling.offsets_t(d,sdecompdim),T.shape)

@ti.kernel
def test_SmoothSelling():
    for i in T:
        mi = (1-T[i])*m[0] + T[i]*m[1] # 
        λ[i], e[i]  = Selling.decomp(mi)
        sλ[i],se[i] = Selling.decomp_smooth2(mi)
test_SmoothSelling()

Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
sΛ,sE = Selling.DecompWithFixedOffsets(sλ.to_numpy(),se.to_numpy())    


from agdt.Selling import Smooth

np.random.seed(36)
d = 3
mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)
nT = 50
T = ti.field(float_t,nT); T.from_numpy(np.linspace(0,1,nT).astype(np_float_t))
m = ti.field(mat_t,2); m.from_numpy(pyRandomSym(mat_t.n,shape=(2,)).astype(np_float_t))
λ  = ti.field(Selling.weights_t(d),T.shape)
e  = ti.field(Selling.offsets_t(d),T.shape)
sdecompdim = Selling.decompdim(d,smooth=True)
sλ = ti.field(Selling.weights_t(d,sdecompdim),T.shape)
se = ti.field(Selling.offsets_t(d,sdecompdim),T.shape)

@ti.kernel
def test_SmoothSelling():
    for i in T:
        mi = (1-T[i])*m[0] + T[i]*m[1] # 
        λ[i], e[i]  = Selling.decomp(mi)
        sλ[i],se[i] = Smooth.decomp_smooth(mi)
test_SmoothSelling()

Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
sΛ,sE = Selling.DecompWithFixedOffsets(sλ.to_numpy(),se.to_numpy())   