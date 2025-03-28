import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
from matplotlib import pyplot as plt
import taichi as ti
ti.init(arch=ti.cpu,default_fp=ti.f64,debug=True)
np.set_printoptions(linewidth=2000)

from agd.Eikonal import VoronoiDecomposition
from agdt import Selling
from agd.Metrics.misc import expand_symmetric_matrix

def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M


mat_t = ti.math.mat3
float_t = mat_t.dtype
sSel3 = Selling.mk_SmoothSelling3(mat_t.n,mat_t.dtype)
Reconstruct = Selling.mk_Reconstruct(mat_t.n,mat_t.dtype)

if False:
	print("------- Single matrix test --------")

	m = ti.field(mat_t,tuple())

	# Generate m

	#m = mat((1.2,0.2,0.3),(0.2,1.,-0.1),(0.3,-0.1,0.8)); m_np = m.to_numpy() # Hardcoded
	#np.random.seed(42); m_np = pyRandomSym(3);  # Random
#	m_np = expand_symmetric_matrix([1.22654,-0.217325,1.3738,-0.578292,-0.547712,1.15934]); 
	#m_np = np.array([[ 1.22653529, -1.00920993, -0.43091749], [-1.00920993,  2.16568633,  1.03968147], [-0.43091749,  1.03968147,  1.07302102]])
	#m_np = np.array([[1.159344419587, -0.578292440804, -0.547712422558], [-0.578292440804, 1.226535291770, -0.217325361798], [-0.547712422558, -0.217325361798, 1.373801760968]])
	m_np = np.array([[ 1.64789217, -0.13179826 , 0.15801286],
 [-0.13179826 , 1.20303931 , 0.26033891],
 [ 0.15801286 , 0.26033891 , 1.29102163]])
	m.from_numpy(m_np)

	λ_np,e_np = VoronoiDecomposition(m_np,relax=0.004,smooth=2) 
	#exit(0)
	print(λ_np)
	print(e_np)

	@ti.kernel
	def test_3d():
		for _ in range(1):
			λ,e = sSel3(m[None])
			print(λ)
			print(e)
	#		print(λ)
	#		print(e)
			print(Reconstruct(λ,e)-m[None])
	test_3d()

#	exit(0)

if False:
	print("-------- Multiple matrix test --------")

	np.random.seed(42)
	m_np = pyRandomSym(3,shape=(5,))
	print(m_np[0])

	λ_np,e_np = VoronoiDecomposition(np.moveaxis(m_np,0,-1),smooth=2) 
	#print("First decomp, np",λ_np[:,0],e_np[:,:,0])
	#print("First decomp, np",λ_np[0],e_np[0])

	print("shapes",λ_np.shape,e_np.shape)
	λ_np = λ_np.T 
	e_np = e_np.T
	Λ_np,E_np = Selling.DecompWithFixedOffsets(λ_np,e_np)
	#print("shapes",λ_np.shape,e_np.shape)

	m = ti.field(mat_t,m_np.shape[:-2]); m.from_numpy(m_np)
	λ = ti.field(sSel3.types.weights_t,m.shape)
	e = ti.field(sSel3.types.offsets_t,m.shape)
	m_rec = ti.field(mat_t,m.shape)

	@ti.kernel
	def test_multiple():
		for I in m:
			λ[I],e[I] = sSel3(m[I])
			m_rec[I] = Reconstruct(λ[I],e[I])
	test_multiple()

	assert np.allclose(m.to_numpy(),m_rec.to_numpy())
	#print("First decomp, ti",λ[0],e[0])

	#exit(0)

	Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
	assert np.allclose(Λ,Λ_np)
	assert np.allclose(E,E_np) 


print("---------- Plotting ------------------")


np.random.seed(36)
#mat_t = ti.lang.matrix.MatrixType(3,3,2,float_t)
T_np = np.linspace(0,1,50) # Issue with 14
nT = T_np.size
T = ti.field(float_t,nT); T.from_numpy(T_np)
m = ti.field(mat_t,2); m.from_numpy(pyRandomSym(mat_t.n,shape=(2,)))
print((1-T[14])*m[0]+T[14]*m[1])

Sel = Selling.mk_Selling(mat_t.n,mat_t.dtype)
sSel = Selling.mk_SmoothSelling3(mat_t.n,mat_t.dtype,relax=0.04)

λ  = ti.field( Sel.types.weights_t,T.shape)
e  = ti.field( Sel.types.offsets_t,T.shape)
sλ = ti.field(sSel.types.weights_t,T.shape)
se = ti.field(sSel.types.offsets_t,T.shape)

@ti.kernel
def test_SmoothSelling():
    for i in T:
        mi = (1-T[i])*m[0] + T[i]*m[1] # 
        λ[i], e[i]  =  Sel(mi)
        sλ[i],se[i] = sSel(mi)
test_SmoothSelling()

Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())
sΛ,sE = Selling.DecompWithFixedOffsets(sλ.to_numpy(),se.to_numpy())

plt.title("SmoothSelling decomposition of a linear family of matrices")
for λi,ei in zip(sΛ.T,sE):
    plt.plot(T.to_numpy(),λi,label=f"{ei}")
plt.legend()
plt.show()