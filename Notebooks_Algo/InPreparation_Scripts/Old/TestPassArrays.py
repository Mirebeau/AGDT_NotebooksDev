import taichi as ti
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)


@ti.pyfunc
def f3(a,b,c):
	return a+b+c

@ti.pyfunc
def f2(a,b):
	return a+b

#@ti.pyfunc
#def eval_ti(f:ti.template(),*args):
#	return f(*args)

arr1 = ti.field(float_t,shape=(3,)); arr1.fill(1)
arr2 = ti.Vector.field(2,float_t,shape=(3,)); arr2.fill(2)
arr3 = ti.field(float_t,shape=tuple()); arr3.fill(3)
data = (arr1,arr2)

@ti.pyfunc
def get(a,x):
	if ti.static(a.shape==tuple()): return a[None]
	else: return a[x]
#		for i in ti.static(range(x.n)):
#			if a.shape[i]==1: x[i]=0
#		return a[x]

@ti.pyfunc
def g(x,arr1_,arr2):
	arr1= get(arr1_,x)
	return x+arr1+arr2[x][1]

print(arr1.shape)

@ti.kernel
def test(dat:ti.template()):
	for x in range(3):
#		print(g(x,arr1,arr2))
		print(g(x,*dat))
#		print(f2(x,x))
#		print(f2(*(x,x)))
#		print(f3(x,*(x,x)))

test(data)


arr = ti.field(float_t,shape=tuple()); arr.fill(0); #arr[0,0]=1
shape=(2,2)

@ti.pyfunc
def get2(a,x):
	if ti.static(a.shape==tuple()): return a[None]
	for i in ti.static(range(x.n)):
		if a.shape[i]==1: x[i]=0
	return a[x]


@ti.kernel
def test2():
	for x in ti.grouped(ti.ndrange(*shape)):
		print(x,get2(arr,x))
test2()

arrVec = ti.field(dtype=ti.math.vec2,shape=())
arrVec.fill(1); print(arrVec)
arrVec.fill([2,3]); print(arrVec)

print(arrVec.dtype,ti.math.vec2.dtype,arrVec.m,arr.m)