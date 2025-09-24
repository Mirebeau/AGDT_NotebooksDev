import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

import numpy as np
import taichi as ti
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)
np.set_printoptions(linewidth=2000)

from agdt import AD

#fwd_t = AD.mk_fwd1(1,float_t)
fwd_t = AD.fwd0 #mk_fwd0(1,float_t)


#x0_ad = fwd_t(2.,(1.,)
@ti.kernel
def test_fwd():
	x_ad = fwd_t.types.mk(2.,0)

#	x_ad = x0_ad
	for _ in range(1):
#		x_ad.iadd(x_ad)
		y_ad = x_ad.add(x_ad)
		y_ad.print()
		y_ad.iadd(x_ad)
		y_ad.print()
		x_ad.print()
		y_ad.print()

		y_ad.imulc(2.)
		y_ad.print()
		y_ad.isubc(3.)
		y_ad.print()

		z = fwd_t.types.mk(3.,0)
		z.print()

test_fwd()


fwd1 = AD.mk_fwd1(2)


to_fwd = AD.fwd_translator()
@to_fwd
def f(x,y): return 2*x+y**2
print(f( fwd1(3,[4,5]), fwd1(2,[0,1]) ))
print(f.orig( 3, 2 ))

@to_fwd
def g(x): return np.log(2-x)
print(g(fwd1(1,[4,5])))
print(g.orig(1))
exit(0)

# def f2(x,y): return x+y
# import inspect
# print("signature",inspect.signature(f2))
# print(type(inspect.signature(f2)))

# @ti.pyfunc
# def f(x,y):
# 	return x+y

# def g(x): return f(x,x)
# g = ti.pyfunc(g)

# print(inspect.signature(g))

# @ti.kernel
# def test_vararg():
# 	for _ in range(1):
# 		x=3
# 		print(f(2,2))
# 		print(f(*(x,2)))
# 		print(g(x))
# test_vararg()




def mk_fwdfun(n):
	# Taichi does not accept variable length arguments, so I must turn to dreaded "exec" codegen 
	# (Taichi also does not accept lambdas)
	assert isinstance(n,int) and n<=32
	args = ",".join([f"x{i}" for i in range(n)])
	fwdfun_ = f"""
#---------------------------------------------------------------
class fwdfun{n}:
	def __init__(self,f):
		self.f = ti.pyfunc(f)
	def __add__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).add(other.f({args}))
		else:
			def f({args}): return self.f({args}).addc(other)
		return fwdfun{n}(f)
	def __sub__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).sub(other.f({args}))
		else:
			def f({args}): return self.f({args}).subc(other)
		return fwdfun{n}(f)
	def __mul__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).mul(other.f({args}))
		else:
			def f({args}): return self.f({args}).mulc(other)
		return fwdfun{n}(f)
	def __truediv__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).div(other.f({args}))
		else:
			def f({args}): return self.f({args}).divc(other)
		return fwdfun{n}(f)

	def __iadd__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).iadd(other.f({args}))
		else:
			def f({args}): return self.f({args}).iaddc(other)
		return fwdfun{n}(f)
	def __isub__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).isub(other.f({args}))
		else:
			def f({args}): return self.f({args}).isubc(other)
		return fwdfun{n}(f)
	def __imul__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).imul(other.f({args}))
		else:
			def f({args}): return self.f({args}).imulc(other)
		return fwdfun{n}(f)
	def __itruediv__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).idiv(other.f({args}))
		else:
			def f({args}): return self.f({args}).idivc(other)
		return fwdfun{n}(f)

	def __radd__(self,other):    
		def f({args}): return self.f({args}).addc(other)
		return fwdfun{n}(f)
	def __rmul__(self,other):    
		def f({args}): return self.f({args}).mulc(other)
		return fwdfun{n}(f)
	def __rsub__(self,other):    
		def f({args}): return self.f({args}).rsubc(other)
		return fwdfun{n}(f)
	def __rtruediv__(self,other):    
		def f({args}): return self.f({args}).rdivc(other)
		return fwdfun{n}(f)
	def __pow__(self,other):    
		def f({args}): return self.f({args}).powc(other)
		return fwdfun{n}(f)
	def __neg__(self,other):    
		def f({args}): return self.f({args}).neg(other)
		return fwdfun{n}(f)

	def log(self):
		def f({args}): return self.f({args}).log()
		return fwdfun{n}(f)
	def exp(self):
		def f({args}): return self.f({args}).exp()
		return fwdfun{n}(f)
	def abs(self):
		def f({args}): return self.f({args}).abs()
		return fwdfun{n}(f)
	def sin(self):
		def f({args}): return self.f({args}).sin()
		return fwdfun{n}(f)
	def cos(self):
		def f({args}): return self.f({args}).cos()
		return fwdfun{n}(f)
	def tan(self):
		def f({args}): return self.f({args}).tan()
		return fwdfun{n}(f)
	def arctan(self):
		def f({args}): return self.f({args}).arctan()
		return fwdfun{n}(f)
	"""
	idnk_ = [f"""def id{n}{k}({args}):return x{k}""" for k in range(n)]
	idn_ = f"id{n} = ["+",".join([f"fwdfun{n}(id{n}{k})" for k in range(n)])+"]"
	dec_ = f"dec{n} = lambda f:f(*id{n}).f"
	return "\n".join([fwdfun_]+idnk_+[idn_,dec_])


print( mk_fwdfun(1) )
print( mk_fwdfun(2) )
print( mk_fwdfun(3) )

exit(0)

#d={}; exec(mk_fwdfun(1),d)


dec1 = AD.mk_dec1()
fwd1 = AD.mk_fwd1(2)


def f(x): return x+x+1
f_fwd = dec1(f)

print( f_fwd(AD.fwd0(2)) )
print( f_fwd(fwd1(3,[4,5])))

@ti.kernel
def testexec():
	x0 = AD.fwd0(2)
	x2 = fwd1(3,[4,5])
	print(f_fwd(x0).x)
	print(f_fwd(x2).v)
testexec()


exit(0)

fwdfun1 = lambda f : f(*d['idn']).f


#fwdfun2 = mk_fwdfun(2)

#def fwdfun1():


#	return lambda f : f(*id{n}).f


exit(0)

class _fwdfun1:
	def __init__(self,f):
		self.f = ti.pyfunc(f)
	# def __add__(self,other):
	# 	if isinstance(other,_fwdfun1):
	# 		def f(x): return self.f(x).add(other.f(x))
	# 		return _fwdfun1(f)
	# 	def f(x): return self.f(x).addc(other.f(x))
	# 	return _fwdfun1(f)

	def __add__(self,other):
		if isinstance(other,_fwdfun1): 
			def f(x): return self.f(x).add(other.f(x))
		else:
			def f(x): return self.f(x).addc(other)
		return  _fwdfun1(f)

fwdfun1 = mk_fwdfun(1)

def f(x): return x+x+1
def id1(x): return x
f_fwd = f(_fwdfun1(id1)).f

f_fwd1 = f(_fwdfun1(id1)).f
print( f_fwd(AD.fwd0(2)) )
print( f_fwd1(AD.fwd0(2)) )

@ti.kernel
def test1():
	for _ in range(1):
		x0 = AD.fwd0(2)
		print(f_fwd(x0).x)
		print(f_fwd1(x0).x)

test1()

# Now implent an iteration for some GSD-like problem

# def f(x): return np.log(x*x+2)
# f_fwd = AD.fwdfun(f,1)
# print(f(2.))
# print(f_fwd(AD.fwd0(2.)))
# fwd1 = AD.mk_fwd1(2)
# print(f_fwd(fwd1(2.,[4,5])))

# print(fwd1.types.mk(2,1))


# @ti.kernel
# def test_fwdfun():
# 	print(f_fwd(fwd1(2.,[4,5])))
#test_fwdfun()






# def f(x): return x*x+x
# f_fwd = fwdfun(f,1)

# x0 = AD.fwd0(2)
# print(x0)
# print(x0.add(x0))
# print(x0)

# print(f_fwd(x0))

# def g(x): return np.log(x)
# g_fwd = fwd_translate(g,1)
# print(g_fwd(x0))



# from taichi.lang.ops import is_taichi_expr

# @ti.pyfunc
# def print_me(x):
# 	if is_taichi_expr(x): print("x is taichi")
# 	print("x is python")

# print_me(x0)

# @ti.kernel
# def test_print():
# 	for _ in range(1):
# 		x0 = AD.fwd0(2)
# 		print_me(x0)
# test_print()

# import copy
# x1 = AD.fwd0(x0.x)
# #x1 = copy.deepcopy(x0)
# print(x1)
# x1.x+=2
# print(x0)
# print(x1)

# fwd1 = AD.mk_fwd1(2)
# x2 = fwd1(3,[4,5])
# print(x2)
# x3 = x2.copy()
# x3.iadd(x2)
# print(x3)
# print(x2)

