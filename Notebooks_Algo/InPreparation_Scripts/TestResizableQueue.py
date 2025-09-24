import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
from agdt.Eikonal import Queue,CappedQueue
from agdt.Eikonal.Old import Queue as OldQueue
#from agdt.Eikonal import Metrics,HFM
np.set_printoptions(linewidth=2000)
ti.init(arch=ti.cpu, debug=True)

class dummy_class:pass
testrecomp = dummy_class()
testrecomp.x = 0

# ------------- Priority Queue ------------
if False:
    pq = Queue.priority_queue.init(ti.f32,ti.i32,capacity=6)
    @ti.pyfunc
    def testpq(self:ti.template()):
        print("testpq")
        for prio,elem in ti.static(((2.5,1),(3.2,2),(1.5,3))): pq.push(self,prio,elem)
        print(pq.top(self))
        print(pq.empty(self))
        pq.pop(self)
        print(pq.top(self))
        pq.clear(self)
        print(pq.empty(self))
        print(pq.capacity(self))
        print(pq.size(self))

    print("--- Python scope ---")
    testpq(pq)
    pq = pq.with_capacity(pq)
    testpq(pq)

    print("Taichi scope")
    @ti.kernel
    def testpq_ker(self:pq.argtype):
        testpq(self)
        print(f"{testrecomp.x=}")

    #    testpq(pq_ti)
    testpq_ker(pq)
    pq = pq.with_capacity(pq)
    testpq_ker(pq)

if False:
    pq = CappedQueue.priority_queue(ti.f32,ti.i32,capacity=6)

    @ti.pyfunc
    def testpq():
        print("testpq")
        for prio,elem in ti.static(((2.5,1),(3.2,2),(1.5,3))): pq.push(prio,elem)
        print(pq.top())
        print(pq.empty())
        pq.pop()
        print(pq.top())
        pq.clear()
        print(pq.empty())
        print(pq.capacity())
        print(pq.size())

    print("--- Python scope ---")
    testpq()

    print("--- Taichi scope ---")
    @ti.kernel
    def testpq_ker():
        testpq()
        print(f"{testrecomp.x=}")
    testpq_ker()

if False:
    fifo = Queue.fifo.init(ti.f32,capacity=6)

    @ti.pyfunc
    def testfifo(self:ti.template()):
        for x in ti.static((0.5,1.2,-0.1)): fifo.push(self,x)
        print(fifo.front(self))
        fifo.pop(self)
        print(fifo.front(self))
        print(fifo.size(self))
        print(fifo.empty(self))
        print(fifo.capacity(self))
    
    print("--- Python scope ---")
    testfifo(fifo)
    fifo = fifo.with_capacity(fifo)
    testfifo(fifo)
    print("--- Taichi scope ---")
    @ti.kernel
    def testfifo_ker(self_fifo:fifo.argtype):
        testfifo(self_fifo)
        print(f"{testrecomp.x=}")
    testfifo_ker(fifo); testrecomp.x=1
    fifo = fifo.with_capacity(fifo)
    testfifo_ker(fifo)

# ---------------- Fifo -----------------
if False:
    fifo = Queue.fifo.init(ti.f32,capacity=6)

    @ti.pyfunc
    def testfifo(self:ti.template()):
        for x in ti.static((0.5,1.2,-0.1)): fifo.push(self,x)
        print(fifo.front(self))
        fifo.pop(self)
        print(fifo.front(self))
        print(fifo.size(self))
        print(fifo.empty(self))
        print(fifo.capacity(self))
    
    print("--- Python scope ---")
    testfifo(fifo)
    fifo = fifo.with_capacity(fifo)
    testfifo(fifo)
    print("--- Taichi scope ---")
    @ti.kernel
    def testfifo_ker(self_fifo:fifo.argtype):
        testfifo(self_fifo)
        print(f"{testrecomp.x=}")
    testfifo_ker(fifo); testrecomp.x=1
    fifo = fifo.with_capacity(fifo)
    testfifo_ker(fifo)

if False:
    fifo = CappedQueue.fifo(ti.f32,capacity=6)
    @ti.pyfunc
    def testfifo():
        for x in ti.static((0.5,1.2,-0.1)): fifo.push(x)
        print(fifo.front())
        fifo.pop()
        print(fifo.front())
        print(fifo.size())
        print(fifo.empty())
        print(fifo.capacity())
    
    print("--- Python scope ---")
    testfifo()
    print("--- Taichi scope ---")
    @ti.kernel
    def testfifo_ker():
        testfifo()
        print(f"{testrecomp.x=}")
    testfifo_ker()

# ------------- Lifo -------------

if  False:
    lifo = Queue.lifo.init(ti.i32,capacity=6)
    @ti.pyfunc
    def testlifo(self:ti.template()):
        for x in ti.static((3,1,2)): lifo.push(self,x)
        print(lifo.top(self))
        lifo.pop(self)
        print(lifo.top(self))
        print(lifo.size(self))
        print(lifo.capacity(self))
        print(lifo.empty(self))
    print("--- Python scope ---")
    testlifo(lifo)
    print("--- Taichi scope ---")
    @ti.kernel
    def testlifo_ker(self_lifo:lifo.argtype):
        testlifo(self_lifo)
    testlifo_ker(lifo)

if True:
    lifo = CappedQueue.lifo(ti.i32,capacity=6)
    @ti.pyfunc
    def testlifo():
        for x in ti.static((3,1,2)): lifo.push(x)
        print(lifo.top())
        lifo.pop()
        print(lifo.top())
        print(lifo.size())
        print(lifo.capacity())
        print(lifo.empty())
    print("--- Python scope ---")
    testlifo()
    print("--- Taichi scope ---")
    @ti.kernel
    def testlifo_ker():
        testlifo()
    testlifo_ker()

exit(0)


float_t = ti.f32; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)

a = ti.ndarray(float_t,shape=10)
a.fill(1)
b = ti.ndarray(float_t,shape=10)
b.fill(2)

view_params_tmpl = ti.types.argpack(view_mtx=ti.types.ndarray(float_t,1), far=ti.f32)
view_params = view_params_tmpl(view_mtx=a,far=1)
@ti.kernel
def p(view_params: view_params_tmpl) -> ti.f32:
    return view_params.far
print(p(view_params)) 

exit(0)


# a_arr = ti.ndarray(float_t,shape=10); a_arr.fill(3)
# b_arr = ti.ndarray(float_t,shape=10); b_arr.fill(4)
# class testclass:pass
# test = testclass
# test.x = a


print("Testing argpack")
prms_tmpl = ti.types.argpack(u=ti.types.ndarray(float_t,1),v=ti.types.ndarray(float_t,1))

@ti.kernel
def testargpack(prms:prms_tmpl):

    print("hello")

my_prms = prms_tmpl(u=a,v=b)
print(my_prms.v[0])
testargpack(my_prms)

exit(0)



@ti.func
def bli():
    print(test.x[1])

@ti.kernel
def bla(x:ti.types.ndarray(float_t,1)):
    print(test.x[0])
    bli()

bla(a_arr) # Changing a field triggers recompilation
test.x = b
bla(b_arr)


exit(0)


@ti.kernel
def printme1(a:ti.template()):
    for _ in range(1):
        print(a[0])


printme1(a)
printme1(b)

print("------- What about tuple ? --------")

@ti.func
def printme2(u,v):
    print(u[0],v[0])

class testclass:pass
test = testclass
test.a=a

@ti.kernel
def printme3(data:ti.template()):
    printme2(*data)
    print(test.a[0])

printme3((a,b))
test.a=b
printme3((a,b)) # Recompilation if I change a,b to b,a

@ti.kernel
def printme3b(data1:ti.template(),data2:ti.template()):
    print(data1[0],data2[0])
    print(test.a[0])

print("Only changing the argument order")
printme3b(a,b)
test.a = a
printme3b(a,b) 
printme3b(b,a) # Changing the argument order triggers a recompilation, even if arrays are passed

# Conclusion : it seems impossible to avoid recompilation if we change the array sizes.
# That is a bit unfortunate. Hopefully, the recompilations will be cheap ??? 
# (Maybe most work is alread done and reusable ?)
# Only need : pass the queue as argument.

exit(0)


print("---------- Namedtuple ---------")

from collections import namedtuple
Data = namedtuple("Data",['a','b'])
class myclass(namedtuple("myclass",['x','y'])):
    @ti.pyfunc
    def printsum(self):
        print(self.x[0]+self.y[1])

@ti.data_oriented
class myclass2:
    def __init__(self,x,y):
        self.x = x; self.y=y
    @ti.pyfunc
    def printtwice(self):
        print(2*self.x[0]); print(2*self.y[1])

data2 = myclass2(b,a)
data2.printtwice()


@ti.kernel
def testNamedTuple(data:ti.template(),data_:ti.template()):
    print('testnamedtuple')
    print(data.x[0])
    data.printsum()
    print("As global")
    data2.printtwice()
    print("As argument")
    data_.printtwice()

#testNamedTuple(Data(a,b))

testNamedTuple(myclass(b,a),data2)

c = ti.field(float_t,3); c.fill(3)
data2.x = c
testNamedTuple(myclass(b,a),data2) # Recompilation if I change a,b to b,a...
data2.printtwice()

exit(0)

# exit(0)

# @ti.dataclass
# class printme4:
#     x:float_t
#     y:ti.types.ndarray(float_t,ndim=1) # Seems to fail. 
#     @ti.func
#     def printme5(self):
#         print(self.x)
#         print(self.y[0])

# @ti.kernel
# def printme6(data:ti.template()):
#     data.printme5()

# data = printme4(3.,a)
# #data.y.fill(1)
# printme6(data)

class myQueue:

    @classmethod
    @ti.func
    def printme(cls,a):
        print(a[0]) 

print("-myQueue-")

@ti.kernel
def printme7():
    myQueue.printme(a)
printme7()