import taichi as ti

@ti.kernel
def inc(x:ti.types.ndarray(ndim=1)):
	for i in x: x[i]+=6

def fun(β): inc(β)

@ti.data_oriented
class myclass:
	def __init__(self):pass
	@ti.kernel
	def α(self,x:ti.types.ndarray(ndim=1)):
		for i in x: x[i]+=1
	def fun(self,x): self.α(x)