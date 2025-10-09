import taichi as ti
import numpy as np
ti.init(arch=ti.cpu, debug=True)

compiling = ti.field(ti.i8,tuple())
def ti_debug(): return ti.lang.impl.default_cfg().debug
compiling = ti.field(ti.i8,tuple())

a = ti.field(ti.f32,1003)
compiling = ti.field(ti.i8, shape=tuple())

@ti.kernel
def testrecomp(a:ti.template()):
	compiling[None]=0
	print(a[0])
testrecomp(a)


exit(0)

@ti.pyfunc
def getitem_broadcast(a:ti.template(),x):
	"""
	If a is an ndarray, extract the value at position x, with broadcasting. Otherwise, return a.
	"""
	if ti.static(isinstance(a,(ti.lang.any_array.AnyArray,ti.lang._ndarray.Ndarray))): 
		ti.static_assert(len(a.shape)==x.n)
		for i in ti.static(range(x.n)):
			if a.shape[i]==1: x[i]=0 # Broadcast
		return a[*x]
	else: return a # A single value is passed

pack_t = ti.types.argpack(a=ti.types.ndarray()); a = ti.ndarray(ti.math.ivec2,tuple()); a.fill(2); pack = pack_t(a)
pack2_t = ti.types.argpack(a=ti.math.vec2); pack2 = pack2_t([6,4])

print(pack2)
print(a.shape)


def make_argpack(**kwargs):
	"""
	Create an argument pack type and instance, whose elements may be ndarrays (passed by reference), 
	or low-dimensional variables (passed by value) 
	**kwargs : dictionary of key:(value,dtype), where value is either an ndarray of dtype, or a dtype.
	"""
	pack_t = []
	for key,(value,dtype) in kwargs.items():
		if isinstance(value,ti.lang._ndarray.Ndarray):
			pack_t.append( (key,ti.types.ndarray(dtype,len(value.shape))) )
			assert value.dtype==dtype or value.dtype==dtype.dtype # Check dtype and dimensions
			assert getattr(value,'n',1)==getattr(dtype,'n',1)
			assert getattr(value,'m',1)==getattr(dtype,'m',1)
		else: pack_t.append( (key,dtype) )
	pack_t = ti.types.argpack(**{key:dtype for key,dtype in pack_t})
	return pack_t(*[val for key,(val,dtype) in kwargs.items()]) # Some implicit conversions

print(hasattr(a,'m'))
print(a.n,ti.math.ivec2.m)
print(make_argpack(a=(a,ti.math.ivec2),b=(3,ti.math.vec2)))

print(make_argpack(m=(np.eye(2),ti.math.mat2)))

exit(0)

print("Python")
print(type(a))
print(a[None])
print(getb(pack.a))
print(getb(pack2.a))

print("Taichi")
@ti.kernel
def testgetb(pack:pack_t,pack2:pack2_t):
	compiling[None]=0
	print(getb(pack.a))
	print(getb(pack2.a))
testgetb(pack,pack2)




exit(0)

# @ti.data_oriented
# class myclass:

# 	@staticmethod
# 	def ahem(self):
# 		print("ahem")

# 	@staticmethod
# 	def huhu(self):
# 		_mc.ahem(self)

# 	@ti.kernel
# 	def myker(cls,a:ti.types.ndarray()):
# 		print(a[0])

# _mc = myclass

# test = myclass()
# myclass.huhu(test)

# a = ti.ndarray(ti.i32,3); a.fill(5)
# test.myker(a)

# exit(0)

# These is no way of passing ndarray to kernel without an argument
# We can put everything in an argpack, but it is a bit dubious.
# Likely, I won't have to reshape my seeds too often, and it will not impact too much code.
# -> Keep them in field
#a = ti.ndarray(ti.int32,tuple())
a = np.zeros(3)
argtype = ti.types.argpack(a=ti.types.ndarray())
arg = argtype(a)

print(type(a))
print(ti.types.ndarray())
print(type(arg))

print(f"{hasattr(arg,'keys')=}")
#exit(0)

# @ti.pyfunc
# def mystatic(self:ti.template(),self_ti:ti.template()=True):
# 	if ti.static(self_ti==True): return self.a
# 	else: return ti.static(self.a)

#@ti.pyfunc
def mystatic(x):
	if ti.static(isinstance(x,ti.lang._ndarray.ScalarNdarray)): return x
	else: return ti.static(x)

@ti.pyfunc
def testargfun(self:ti.template(),self_ti:ti.template()=True):
	b = ti.static(self.a)
	# print("Python" if ti.static(self_ti==True) else "Taichi")
	# b = mystatic(self.a)
	#b = self.a # Fails in taichi
	#b = ti.static(self.a) # Fails in python
	#b = self.a if ti.static(self_ti==True) else ti.static(self_ti.a) # Fails in taichi
	#b = ti.select(ti.static(self_ti==True), self.a, ti.static(self_ti.a)) # Fails in python and taichi
	#b = mystatic(self,self_ti)

	# # Works, but we are in a smaller scope, separate for python and taihci
	# if ti.static(self_ti==True): 
	# 	print("Python")
	# 	b = self.a
	# 	print(b[None])
	# else: 
	# 	print("Taichi")
	# 	b = ti.static(self_ti.a)
	# 	print(b[None])

	#print(ti.static(isinstance(testargfun._is_taichi_function,bool)))
	#print(hasattr(testargfun,'__wrapped__'))
	#a = arg.a if ti.static(hasattr(arg,'cast')) else ti.static(arg.a)
	#print(ti.static(type(arg)))
	#for s in ti.static(attrs): print(ti.static(hasattr(arg,s)))
#	if ti.static(hasattr(arg,'__del__')): 
#		print("python")
#		#a = arg.a
##		print('taichi') 
		#a = ti.static(arg.a)
	#a = mystatic(arg.a)
	#a = ti.static(arg.a)
	#print(self.a[None])

testargfun(arg)
#exit(0)
@ti.kernel
def testargkernel(arg:argtype):
	#s = ti.static(arg)
	a0 = ti.static(arg.a)
	arg.a[None]=0
	testargfun(arg,arg)
testargkernel(arg)

exit(0)

testrecomp = ti.field(ti.i8,tuple())
data = ti.field(ti.i32,tuple()); data.fill(5)
data2 = 7
@ti.kernel
def testrecompkernel():
	testrecomp[None] = 0 # Compiling testrecompkernel
	print(data[None])
	print(data2)
	#testrecomp:ti.i8=0; testrecomp+=0
testrecompkernel()


@ti.data_oriented
class myclass:
	def __init__(self,n):
		self.data = ti.field(ti.i32,tuple())
		self.data[None] = n
		self.thing = ti.math.vec2(n)
		self.n = n
		@ti.kernel
		def test():
			if ti.static(ti_debug()): compiling[None]=0 # testmyclass
			print(self.data[None])
			print(self.thing[0])
			#print(self.n)
		test()
myinst = myclass(12)

exit(0)

@ti.data_oriented
class myclass:
	def __init__(self,n):
		self.data = ti.ndarray(int,n)
	
	@staticmethod
	@ti.kernel
	def myker(self:ti.template(),data:ti.types.ndarray()):
		for i in data:
			data[i] += 1
		print(myclass.testrecomp)

@ti.kernel
def myker(self:ti.template(),data:ti.types.ndarray()):
	for i in data:
		data[i] += 1
	print(myclass.testrecomp)

myclass.testrecomp = 'a'
myinst = myclass(3)
myker(myinst,myinst.data)
myclass.testrecomp = 'b'
myker(myinst,myinst.data)
myinst2 = myclass(4)
myker(myinst2,myinst2.data)
