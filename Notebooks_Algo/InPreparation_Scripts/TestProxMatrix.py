import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")


import numpy as np
import taichi as ti
ti.init(arch=ti.cpu,default_fp=ti.f64)

from agdt.Proximal import MatrixPerspective as MP
from agdt import Sort, Linalg, Selling

#sort = Sort.mk_Sort(4)
v = ti.math.vec4(3,1,2,4)
#argsort = Sort.mk_ArgSort(4)

print(tuple(enumerate((3,4))))
dtype = None
default = float
@ti.func
def f(x,dtype:ti.template()): 
	z=dtype(x)
	return dtype(x)
@ti.func
def g(x,dtype:ti.template()=None): return f(x,Linalg.none2(dtype,float))

#@ti.func
#def f(x,dtype:ti.template()=float):
#	return x+dtype(2)

@ti.kernel
def testsort():
	gates = ti.static(((1,2),(3,4)))

	for _ in range(1):
		print(Sort.sort(v))
		print(Sort.argsort(v))
		for i,j in ti.static(gates):
			print(i+j)
		for i,jk in ti.static(tuple(enumerate(((3,3),(4,5))))): print(i,jk)
#		print(Linalg.Selling(Linalg.random_sym(3)))
		print(Selling.sabs(0.5,1),Selling.sabs(0.5,2))

		print(Linalg.zero_like(v))
		print(Linalg.zero_like(v[0]))
		print(Linalg.full_like(v,3))
		print(v)

#		x:Linalg.none2(dtype,default) = 3
#		print( f(3,Linalg.none2(dtype,default)))
		print(f(2,float))
		print(f(2,int))
		print(g(2))
		print(g(2,int))

		print(Selling.Selling_smooth2(Selling.random_sym(2)))
		print(Selling.Selling_smooth3(Selling.random_sym(3)))


		b = ti.Vector((1,2),ti.i8)
		a = ti.Matrix(((1,2),(3,4)),ti.i8)
		c = a@b
		c+=100; c+=100
		print(c)		

testsort()
exit(0)



def pyRandomSym(ndim,relax=0.1,shape=tuple()):
    """Generate random symmetric matrices"""
    A = 2*np.random.rand(*shape,ndim,ndim)-1
    M = np.swapaxes(A,-1,-2) @ A
    trM = sum(M[...,i,i] for i in range(ndim))
    M += relax*trM[...,None,None]*np.eye(ndim)
    return M

m = ti.lang.matrix.MatrixType(2,2,2,ti.i8)((1,2),(2,3))
v = ti.lang.matrix.VectorType(2,int)(0)

m3 = ti.lang.matrix.MatrixType(3,3,2,ti.i8)((1,2,3),(2,4,5),(3,5,6))

m4 = ti.lang.matrix.MatrixType(2,2,2,ti.i8)((1,0),(0,2))
v4 = ti.lang.matrix.VectorType(2,ti.i8)((1,2))

@ti.kernel
def testa():
	print(MP.diag2mat(v))
	print(MP.mat2diag(m))
	print(MP.sym2flt(m))
	print(MP.sym2flt(m3))
	print(m4 @ v4)

	a = ti.static(np.int8(2))
	b = a
	b += 200
	print(b)
	v=ti.Vector([0.,1.])
	w = Linalg.perp(v)
	print(v,w)

	m1 = ti.lang.matrix.MatrixType(1,1,2,float)(2)
	print(Linalg._obtuse_superbase1(m1))

	mrand = Linalg.random_sym(2)
	for _ in range(1):
		print(Linalg.obtuse_superbase(mrand))
		print(Linalg.Selling(mrand))

		print(Linalg.Selling(Linalg.random_sym(3)))

	print(ti.Vector([[1,2],[3,4]]))
#	print(ti.Vector(v,v))

	print(ti.Vector([i+j for i,j in ((1,2),(4,0))]))
	print(Linalg.sabs(0.5,1),Linalg.sabs(0.5,2))

testa()
exit(0)


@ti.func
def zero(x:ti.template(),cst=0):
	y=x; y=ti.int8(0); return y

@ti.kernel
def test0():
	diag = MP.mat2diag(m)
	diag[0]+=100
	diag[0]+=100
	print(diag)
	u = zero(m)
	a:ti.i16 = 2
	b = zero(a,100)
	b += zero(a,100); b += zero(a,100)
	print(b)
	print(u)
	print(m.trace())
#	vec_t:ti.template() = ti.lang.matrix.VectorType(2,float)


test0()
exit(0)

def mk_fun(dtype=float):
	@ti.func
	def fun(x): return dtype(x+1)
	return fun

@ti.func
def fun2(x): return x+2


@ti.kernel
def test():
	fun,fun2 = ti.static(mk_fun(ti.int8),fun2)
	A = 0.
	B = fun(A) 
	B+=100
	B+=100
	print(B)
	print(m)
	w = ti.zero(m[0,:])
	print(v-w)
	x = ti.Vector([1.,2.,3.])
	print(x)
	a:ti.int8 = 0
	a += ti.int8(100)
	a += ti.int8(100)
	b = a+200
	c = ti.zero(a) # Looses the type
	c+=100
	c+=100
	print(a)
	y = ti.Vector([a,a,a],ti.int8)
	y += a
	print(y,a+300,b,c)
test()

exit(0)


if True:
	mat = ti.math.mat2
	np.random.seed(42)
	m_np = pyRandomSym(mat.n)
	m_ = ti.field(mat,shape=tuple())
	m_.from_numpy(m_np)
	print(m_)

	print("Eigvals",np.linalg.eigvalsh(m_np))
	λ_np,U_np = np.linalg.eigh(m_np)
	print("NP rec",U_np @ np.diag(λ_np) @ U_np.T)

	@ti.kernel
	def test2():
		for _ in range(1):
			m = m_[None]
			print("m",m)

			# Testing normalization
			t,s,m0 = MP.normalize(m)
			print("m",m)
			print(t,s)
			print(m-s*m0) 
			print(m0.trace())

			# Testing eigvalsh
			λ = MP.eigvalsh(m)
			print("Eigvals",λ)

			# Testing eigvals
			#U = MP._eigh2(m,λ)
			λ,U = MP.eigh(m)
			print(U)
			print(U @ U.transpose())
			print(MP.mat2diag(U))
			print(MP.diag2mat(λ))
			print(U.transpose() @ MP.diag2mat(λ) @ U - m) # Probably bad convention...
	test2()

if False:
	mat = ti.math.mat3
	np.random.seed(42)
	m_np = pyRandomSym(mat.n)
	m_np -= np.eye(3)*sum(m_np[i,i] for i in range(3))/3; m_np/=np.sqrt(np.sum(m_np**2))
	print(np.linalg.det(m_np))
	m_ = ti.field(mat,shape=tuple())
	m_.from_numpy(m_np)
	print(m_)
	ev = np.linalg.eigvalsh(m_np)
	print("Eigvals",ev,np.sum(ev**2))
	λ_np,U_np = np.linalg.eigh(m_np)
	assert np.allclose(m_np, U_np @ np.diag(λ_np) @ U_np.T)

	_alli3_t = ti.lang.matrix.MatrixType(3,3,2,ti.i32)
	_alli3 = _alli3_t((0,1,2),(1,2,0),(2,0,1))
#	print(MP._alli3)

	@ti.kernel
	def test3():
		for _ in range(1):
			m=m_[None]
			# Testing normalization
			#t,s,m0 = MP.normalize(m)
			#print("m",m)
			#print(t,s)
			#print(m-s*m0) 
			#print(m0.trace())

			λ = MP.eigvalsh3(m)
			print("Eigvals",λ)
			print(_alli3)

			U = MP._eigh3(m,λ)
			print(U.transpose() @ MP.diag2mat(λ) @ U - m)
	test3()

if False:
	mat = ti.math.mat4
	np.random.seed(42)
	m_np = pyRandomSym(mat.n)
	m_np -= np.eye(4)*sum(m_np[i,i] for i in range(4))/4; m_np/=np.sqrt(np.sum(m_np**2))
	print(np.linalg.eigvalsh(m_np))
	m_ = ti.field(mat,shape=tuple())
	m_.from_numpy(m_np)

	@ti.kernel
	def test4():
		for _ in range(1):
			m = m_[None]
			λ = MP._eigvalsh4_normalized(m)
			print(λ)

			U = MP._eigh4(m,λ)
			print(U.transpose() @ MP.diag2mat(λ) @ U - m)


	test4()


mat = ti.math.mat2
np.random.seed(42)
m_np = pyRandomSym(mat.n)
m_ = ti.field(mat,shape=tuple())
m_.from_numpy(m_np)

@ti.kernel
def test_prox2():
	for _ in range(1):
		m = m_[None]
		μ = MP.mat1(1)

		MP._prox(μ,M)



