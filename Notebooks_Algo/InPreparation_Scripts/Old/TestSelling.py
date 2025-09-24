import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")


import numpy as np
import taichi as ti
ti.init(arch=ti.cpu,default_fp=ti.f32)

from agdt import Selling


mat = ti.math.mat3
m = mat(
	(1.2,0.2,0.3),
	(0.2,1.,-0.1),
	(0.3,-0.1,0.8))

ObtuseSuperbase = Selling.mk_ObtuseSuperbase(mat.n,mat.dtype)
sel = Selling.mk_Selling(mat.n,mat.dtype)
Reconstruct = Selling.mk_Reconstruct(mat.n,mat.dtype)

sSel3 = Selling.mk_SmoothSelling3(mat.n,mat.dtype)

@ti.kernel
def test_3d():
	for _ in range(1):
		λ,e = sSel3(m)
		print(λ)
		print(e)
		print(Reconstruct(λ,e)-m)
test_3d()
exit(0)

#sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/2024/12_Decembre/BBB_Notebooks_copy")
#import BBB.Prox
#solve3 = BBB.Prox.mk_solve3_perspective()


#sym2 = Selling.mk_Sym(2)

#m = sym2(0)
#print(m) 

#m = ti.math.mat2(0)
#@ti.kernel
#def test0():
#	b = Selling.Superbase(m)
#test0()

mat = ti.math.mat2
m = mat((1.2,0.2),(0.2,1))
ObtuseSuperbase = Selling.mk_ObtuseSuperbase(mat.n,mat.dtype)
sel = Selling.mk_Selling(mat.n,mat.dtype)
Reconstruct = Selling.mk_Reconstruct(mat.n,mat.dtype)

sSel2 = Selling.mk_SmoothSelling2(mat.n,mat.dtype)

@ti.kernel
def test0():
	for _ in range(1):
		b = ObtuseSuperbase(m)
		print(b)
		λ,e = sel(m)
		print(Reconstruct(λ,e))
		μ,f = sSel2(m)
		print(Reconstruct(μ,f))
if True: test0()

exit(0)

#λ_ = λ.to_numpy()
#e_ = e.to_numpy()
#Selling.DecompWithFixedOffsets(λ_,e_)
Λ,E = Selling.DecompWithFixedOffsets(λ.to_numpy(),e.to_numpy())

print(λ,e)
print(Λ,E)

Sabs = Selling.mk_Sabs()
@ti.kernel
def test2():
	for _ in range(1):
		print(Sabs(1.2),Sabs(0.7))
	print(Selling.Smed(0.5,0.8,1.2))
test2()

from agdt import Sort
argsort = Sort.mk_ArgSort(3)
sort = Sort.mk_Sort(3)
x = ti.math.vec3(1,3,2)
@ti.kernel
def test_sort():
	for _ in range(1):
		print(argsort(x),sort(x))
test_sort()



