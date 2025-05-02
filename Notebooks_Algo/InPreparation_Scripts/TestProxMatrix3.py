import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")


import numpy as np
import taichi as ti
ti.init(arch=ti.cpu,default_fp=ti.f64)
#from agdt.Proximal import MatrixPerspective as MP

from agdt.Proximal.sym_eig4 import sym_eig4
from agdt import Linalg
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp


def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M

np.random.seed(42)
m_np = pyRandomSym(4)
m_np=np.array([[1.000000000000, 0.000000000000, -0.088713532730, 0.164044489449], [0.000000000000, 1.000000000000, 0.318703142440, 0.069762083197], [-0.088713532730, 0.318703142440, -1.480868281962, 0.173951068244], [0.164044489449, 0.069762083197, 0.173951068244, -1.236114162825]])
#m_np-=np.eye(4)*lp.trace(m_np)/4; m_np/=np.sqrt(np.sum(m_np**2)) #Normalize

m = ti.Matrix(m_np)

λ_np,e_np = np.linalg.eigh(m_np)
print(λ_np,"\n",e_np)

@ti.kernel
def test_eig4():
	for _ in range(1):
		m0 = -m
		#sym_eig4(m0)
		λ,e = sym_eig4(m0)
		#print(λ)
		#print(e)

		print(e.transpose()@e)
		print(m0-e@Linalg.diag2mat(λ)@e.transpose())
test_eig4()


@ti.kernel
def test_eig3():
	for _ in range(1):
		m3 = m[:3,:3]
		λ,e = ti.sym_eig(m3)
		print(λ)
#		print(e)

		print(e.transpose()@e)
		print(m3-e@Linalg.diag2mat(λ)@e.transpose())
#test_eig3()