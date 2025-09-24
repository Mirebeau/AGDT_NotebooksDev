from collections import namedtuple
#from rcdtype import recordtype

import taichi as ti
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)



@ti.func
def f(x,dtype:ti.template(),dtype2:ti.template()): 
	y:dtype = x
	z:dtype2 = y
	return x*z


@ti.func
def g(x,dtypes:ti.template()):
	y:dtypes.float_t = x
	return y*x

#dtypes = (ti.f32,ti.i32)
HFMtypes = namedtuple('HFMtypes',"float_t int_t")
dtypes = HFMtypes(ti.f32,ti.i32)

#HFMtypes2 = recordtype('HFMtypes2',"float_t int_t")
#dtypes2 = HFMtypes2(ti.f32,ti.i32)

dtypes2 = dtypes._replace(float_t=ti.i64)


arr = ti.field(dtype=ti.f32,shape=(2,2))
vec = ti.math.vec2([4,5.])


@ti.kernel
def test():
	for x in range(1):
		print(f(2,ti.f32,ti.i32))
		print(g(3,dtypes2))
		print(vec)
		arr[0,:]=vec[:]
		print(arr)

test()











