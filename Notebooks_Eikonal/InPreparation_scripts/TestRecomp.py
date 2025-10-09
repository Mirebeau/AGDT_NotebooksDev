"""
We check that not too much is recompiled between runs.
"""
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
from agdt.Eikonal import Metrics,HFM
from agd.Plotting import quiver

from matplotlib import pyplot as plt
import taichi as ti
import numpy as np
np.set_printoptions(linewidth=2000)

float_t = ti.f32; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)
arr_t = ti.types.ndarray()

if True:
    @ti.kernel
    def test_loop():
        for _ in range(1):
            for i in ti.static(range(5)):
                ti.static_print("Hi")
            for i in range(5):
                ti.static_print("Hey")
    test_loop()


if False:
    from agdt.Sort import argsort
    x = ti.lang.matrix.VectorType(9,float_t)([4,2,7,3,1,8,2,5,4])
    r = argsort(x,dtype=float_t)
    for i in range(x.n): print(x[r[i]])
    

if False:
    b = ti.math.ivec2(4,5)
    @ti.kernel
    def testScal():
        a = ti.math.ivec2(2,3)
        print(a @ b) # b is seen as a compile time constant
    testScal()
    b[0]=1
    testScal()

if False:
    arr_pack_t = ti.types.argpack(a=arr_t,b=arr_t)

    @ti.pyfunc
    def gets(arr_pack:ti.template(),ind,name:ti.template()): #:arr_pack_t,ind:ti.template()): #,name:ti.template()):
        if ti.static(isinstance(arr_pack[name],(ti.lang.any_array.AnyArray,ti.lang._ndarray.Ndarray))): 
            return arr_pack[name][ind[ti.static(arr_pack.keys.index(name))]]
        else: return arr_pack[name]
        #print(arr_pack['a'][ind[ti.static(arr_pack.keys.index('a'))]])

        #return pack[name][ind_pack[ti.static(pack.keys.index(name))]]
        
    @ti.kernel
    def extract(arr_pack:arr_pack_t,ind:ti.math.ivec2):
        print(arr_pack['a'][ind[ti.static(arr_pack.keys.index('a'))]])
        print(gets(arr_pack,ind,'a'))

    a = ti.ndarray(float,2); a.fill(1)
    b = ti.ndarray(float,3); b.fill(2)
    extract(arr_pack_t(a,b),ti.math.ivec2([1,2]))
#    val_pack_t = ti.types.argpack(a=float_t,b=int_t)
exit(0)





if False:# Checking declaration of argpack variables in kernels (FAILS)

    arr_pack_t = ti.types.argpack(a=ti.types.ndarray(float_t,1))
    val_pack_t = ti.types.argpack(a=float_t)

    @ti.func
    def extract(arr_pack:ti.template(),x):
        for s in ti.static(arr_pack.keys):
            print(arr_pack[s][0])
        print('Done')
        #return (arr_pack.a[x+s] for s in ti.static((0,1)))

        #return arr_pack.a[x],arr_pack.a[x+1]
        #return val_pack_t(arr_pack.a[x])

    @ti.kernel
    def test_extract(arr_pack:arr_pack_t):
        print("HI",arr_pack.a[0])
        #val_pack = val_pack_t(arr_pack.a[0])
        val_pack = val_pack_t
        extract(arr_pack,1)
        print('DDone')

    a = ti.ndarray(float_t,3); a.from_numpy(np.array([1,2,3]))
    test_extract(arr_pack_t(a))

if False:
    @ti.data_oriented
    class myclass:
        def __init__(self):
            self.a = ti.ndarray(float_t,4); self.a.fill(5)
            self.b = ti.math.vec2([4,5])

        @ti.pyfunc
        def print(self:ti.template()):
            #a = ti.static(self.a)
            print(self.b[1])

        def run_ker(self):
            @ti.kernel
            def test():
                self.print()
            test()
        
        @ti.kernel
        def run_ker2(self):
            #a = ti.static(self.a)
            print(self.b[0])

    myinstance = myclass()
    myinstance.print()
    myinstance.run_ker()
    myinstance.b[1]=9
    myinstance.run_ker()
    myinstance.run_ker2()
    exit(0)



if False:
    arr = ti.ndarray(float_t,tuple()); arr.fill(3) #arr[0]=1; arr[1]=2
    brr = ti.ndarray(float_t,4); brr.fill(5)
    crr = ti.ndarray(float_t,7); crr.fill(7)

    name='init'
    @ti.pyfunc
    def funtest_arr(a:ti.template()):
        #if ti.static(len(a.shape)==0): print("Singleton")
        if ti.static(len(a.shape)==0): print("Singleton")
        else: print("Positive dimension")
            #if ti.static(a.shape[0]==5): print("gran") # Fails (good)
            #for i in ti.static(a.shape[0]): print("Hello") # Fails (good)
        print(a.shape)
        print(name)

    @ti.kernel
    def test_arr(a:ti.types.ndarray()):
        #print(a[None])
        #print(a[0])
        funtest_arr(a)
    name='arr'; test_arr(arr)
    name='brr'; test_arr(brr)
    name='crr'; test_arr(crr)
    name='drr'; test_arr(arr)

    exit(0)

# Let us first build a domain, the rectangle [-1,1]x[0,1] discretized on a grid, and choose a metric type
bounds = [[0,1],[0,1]]
dimx = 5
dom = HFM.Domain(bounds,(dimx,dimx), metric = Metrics.Diagonal(2,float_t))
X = dom.grid()

dom.build_scheme()