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
pm = pyRandomSym(3,shape=(2,))
pm = pm[0] #pyRandomSym(3)
#print(pm)
#exit(0)

m = ti.field(ti.lang.matrix.MatrixType(3,3,2,float_t),tuple())
m.from_numpy(pm)
pλ,pe = VoronoiDecomposition(pm,smooth=2)

olddecomp_smooth = OldImplem_Selling.mk_SmoothSelling3(3)


@ti.kernel
def test():
    for _ in range(1):
        #for i in range(3):
        #    for j in range(3): 
                #print(i,j)
        #        m[i,j]=pm[i,j]
        print("m",m[None])
        m0 = m[None]
#        Smooth.decomp_smooth3(m0)

        λ,e = Smooth.decomp_smooth3(m0)
        print("new rec",m0-Selling.reconstruct(λ,e))
        print("---old version---")
#        λ,e = olddecomp_smooth(m0)
#        print("old rec",m0-Selling.reconstruct(λ,e))

test()
"""
        λ,e = Smooth.decomp_smooth3(m0)
        m_rec = Selling.reconstruct(λ,e)
        m_diff = m0-m_rec 
        #print("m_rec",m_rec)
#        print("Sel",max(abs(Selling.reconstruct(λ,e)-m[None])))
        #print("diff",m_diff,m0[0,0],m_rec[0,0],m_diff[0,0])


        oλ,oe = olddecomp_smooth(m0)
        print(oλ,λ)

test()
"""