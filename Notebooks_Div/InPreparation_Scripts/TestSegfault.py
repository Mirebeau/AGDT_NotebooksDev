import taichi as ti
import numpy as np
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t) #,debug=True)

def f():
	x = ti.field(float_t,100)
	x.fill(0)
	return x

# Segfault. Visiblement, mon install est foireuse
for i in range(100):
	assert np.allclose(f().to_numpy(), f().to_numpy())
