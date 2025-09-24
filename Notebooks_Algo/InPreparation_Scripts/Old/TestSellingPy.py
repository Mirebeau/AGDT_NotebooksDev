import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")

import taichi as ti
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)

from agdt import Selling

ndim=2
m = ti.math.mat2([[1,0.5],[0.5,2]])

@ti.kernel
def test():
	for _ in range(1):
#		m = Selling.random_sym(2)
		λ,e = Selling.decomp(m)
		print(λ)
		print(m)

test()
print(m)
#print(Selling.decomp(m))
#print(ti.random())
#print(Selling._obtuse_superbase2(m))
print(m@m@m)

b = Selling.superbase_t(2)((1,0),(0,1),(-1,-1)) # Canonical superbase

print(b,type(b),b[1,:],type(b[1,:]))
print(b[1,:] @ m)