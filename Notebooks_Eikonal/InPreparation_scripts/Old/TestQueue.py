import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
import agdt
from agdt.Eikonal import Queue

ti.init(ti.cpu)

if False: # Testing functional programming
    @ti.func
    def func(x):
        print(x)

    @ti.func
    def func2(x,callback:ti.template()):
        callback(x)
        callback(x)


    @ti.kernel
    def kernel(callback:ti.template()):
        callback(7)
        func2(5,callback)

    kernel(func)

if False:
    arr = ti.field(dtype=ti.i32,shape=10)
    print(arr.shape)
    print(arr.shape[0])

if False:
    pq = Queue.priority_queue(ti.f32,ti.i32)
    pq.clear()
    print(pq.empty(),pq.size)
    @ti.kernel
    def test_pq():
        for _ in range(1):
            pq.push(0.,1)
            pq.push(1.,2)
            pq.push(4.,0)
            pq.push(-1.,-2)
            pq.push(0.5,3)
            for i in range(3):
                print(pq.top(),pq.size)
                pq.pop()
            print(pq.empty())     
    test_pq()
    

    pq.set_capacity()

    @ti.kernel
    def test_pq2():
        for _ in range(1):
            for i in range(2):
                print(pq.top(),pq.size)
                pq.pop()
    test_pq2()

    pq = Queue.priority_queue(ti.f32,ti.math.ivec2)
    #pq.swap(1,2)
    pq.push(0.,ti.math.ivec2(1,1))
    print(pq.top())
    pq.pop()
    print(pq.size)

if False:
    fifo = Queue.queue(ti.i32,1024)

    @ti.kernel
    def test_fifo():
        for _ in range(1):
            fifo.push(2)
            fifo.push(1)
            fifo.push(3)
            fifo.push(4)
            fifo.push(5)
            for i in range(3):
                print(fifo.front())
                fifo.pop()
    test_fifo()

    fifo.set_capacity()

    @ti.kernel
    def test_fifo2():
        for _ in range(1):
            for i in range(2):
                print(fifo.front())
                fifo.pop()
    test_fifo2()

@ti.data_oriented
class Graph:
    def __init__(self,values):
        self.values = values
        self.bc = ti.field(dtype=ti.i8,shape=values.shape)
        self.bc.from_numpy(self.values.to_numpy()<np.inf)
    
    @property
    def ndim(self): return len(self.values.shape)

    @ti.func
    def update(self,x):
        newval = np.inf # TODO : make sure type is correct...
        if self.bc[x]: newval = self.values[x]
        else:
            for i in ti.static(range(self.ndim)):
                if x[i]>0: y=x; y[i]-=1; newval = min(newval,1+self.values[y])
                if x[i]<self.values.shape[i]-1: y=x; y[i]+=1; newval = min(newval,1+self.values[y])
        return newval
    
    @ti.func
    def neigh(self,x,callback:ti.template()):
        for i in ti.static(range(self.ndim)):
            if x[i]>0: y=x; y[i]-=1; callback(y)
            if x[i]<self.values.shape[i]-1: y=x; y[i]+=1; callback(y)

    
values = ti.field(dtype=ti.f32,shape=(7,7))

values.fill(np.inf)
seed = ti.math.ivec2((3,3))
values[*seed] = 0

graph = Graph(values)
if False: # Testing update operator
    @ti.kernel
    def test_graph():
        for _ in range(1):
            x = ti.math.ivec2((3,4))
            v = graph.update(x)
            print(v)
    test_graph()

if False: # Testing Dijkstra
    pq = Queue.priority_queue(ti.f32,ti.math.ivec2)
    frozen = ti.field(dtype=ti.i8,shape=values.shape); frozen.fill(False)
    @ti.func
    def dijkstra_update_and_push(y):
        if not frozen[y]: 
            graph.values[y] = graph.update(y)
            pq.push(-graph.values[y],y)

    @ti.kernel # TODO : relaunch kernel with increased capacity if needed
    def test_dijkstra():
        for _ in range(1):
            pq.push(-values[seed],seed) # Can be inside or outside kernel
            while not pq.empty() and pq.size < pq.capacity-2*graph.ndim:
                mval,x = pq.top()
                pq.pop()
                frozen[x] = True
                if graph.values[x]!=-mval: continue
                graph.neigh(x,dijkstra_update_and_push)
    test_dijkstra()

    print(pq.prio.to_numpy()[:10])
    print(pq.elem.to_numpy()[:10])
    print(pq.size)

    print(values)

# Testing AGSI
fifo = Queue.fifo(ti.math.ivec2)
fifo.push(seed+(0,1)) # Should insert all neighbors of the seeds
infifo = ti.field(dtype=ti.i8,shape=values.shape); infifo.fill(False)
@ti.func
def agsi_push(y):
    if not infifo[y]: fifo.push(y); infifo[y]=True

@ti.kernel
def test_agsi(callback:ti.template()):
    for _ in range(1):
        while not fifo.empty() and fifo.size < fifo.capacity-2*graph.ndim:
            x = fifo.front()
            fifo.pop()
            infifo[x] = False
            updt = graph.update(x)
            if updt >= graph.values[x]: continue
            graph.values[x] = updt
            callback(x)
            graph.neigh(x,agsi_push)

latest = ti.field(ti.math.ivec2,shape=())
@ti.func
def mycallback(x):
    latest[None] = x

test_agsi(mycallback)
print(f"{latest[None]=}")
print(fifo.size)
print(fifo.elem.to_numpy()[:10])
print(values)