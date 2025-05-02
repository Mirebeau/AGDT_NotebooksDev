import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

import agdt
from agdt import Selling
import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
from agd.Eikonal import VoronoiDecomposition

float_t = ti.f64
np_float_t = agdt.convert_dtype['np'][float_t]
ti.init(arch=ti.cpu,default_fp=float_t)

import OldImplem_Selling

def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M


from agdt.Selling import Smooth

np.random.seed(36)
d = 3
mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)
nT = 50
T = ti.field(float_t,nT); T.from_numpy(np.linspace(0,1,nT).astype(np_float_t))
m = ti.field(mat_t,2); m.from_numpy(pyRandomSym(mat_t.n,shape=(2,)).astype(np_float_t))

print(m)

ms = ti.field(mat_t,T.shape)
@ti.kernel
def test_Interpolate():
    for i in T: ms[i] = (1-T[i])*m[0] + T[i]*m[1] 
test_Interpolate()

pλ,pe = VoronoiDecomposition(np.moveaxis(ms.to_numpy(),0,-1),smooth=2)
pλ=pλ.T; pe=pe.T
pΛ,pE = Selling.DecompWithFixedOffsets(pλ,pe)   

print(pΛ.shape,pE.shape)

plt.title("(c++ version) SmoothSelling decomposition of a linear family of matrices")
for λi,ei in zip(pΛ.T,pE):
    plt.plot(T.to_numpy(),λi,label=f"{ei}")
plt.legend();
plt.show()

λ  = ti.field(Selling.weights_t(d),T.shape)
e  = ti.field(Selling.offsets_t(d),T.shape)
sdecompdim = Selling.decompdim(d,smooth=True)
sλ = ti.field(Selling.weights_t(d,sdecompdim),T.shape)
se = ti.field(Selling.offsets_t(d,sdecompdim),T.shape)

osλ = ti.field(Selling.weights_t(d,sdecompdim),T.shape)
ose = ti.field(Selling.offsets_t(d,sdecompdim),T.shape)

rec = ti.field(mat_t,T.shape)
orec = ti.field(mat_t,T.shape)

olddecomp_smooth = OldImplem_Selling.mk_SmoothSelling3(3)


@ti.kernel
def test_SmoothSelling():
    for _  in range(1):
        for i in range(T.shape[0]):
            mi = (1-T[i])*m[0] + T[i]*m[1] 
            λ[i], e[i]  = Selling.decomp(mi)
            sλ[i],se[i] = Smooth.decomp_smooth3(mi)
            osλ[i],ose[i] = olddecomp_smooth(mi)

            print("Sel",max(abs(Selling.reconstruct(λ[i],e[i])-mi)))
            print("smo",max(abs(Selling.reconstruct(sλ[i],se[i])-mi)))
            print("osm",max(abs(Selling.reconstruct(osλ[i],ose[i])-mi)))

        #assert max(abs(Selling.reconstruct(λ[i],e[i])-mi))<1e-4
        #assert max(abs(Selling.reconstruct(sλ[i],se[i])-mi))<1e-4
        #assert max(abs(Selling.reconstruct(osλ[i],ose[i])-mi))<1e-4
test_SmoothSelling()

Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
sΛ,sE = Selling.DecompWithFixedOffsets(sλ.to_numpy(),se.to_numpy())   
osΛ,osE = Selling.DecompWithFixedOffsets(osλ.to_numpy(),ose.to_numpy())   


plt.title("SmoothSelling decomposition of a linear family of matrices")
for λi,ei in zip(sΛ.T,sE):
    plt.plot(T.to_numpy(),λi,label=f"{ei}")
plt.legend();
plt.show()


plt.title("(old version) SmoothSelling decomposition of a linear family of matrices")
for λi,ei in zip(osΛ.T,osE):
   plt.plot(T.to_numpy(),λi,label=f"{ei}")
plt.legend();
plt.show()


