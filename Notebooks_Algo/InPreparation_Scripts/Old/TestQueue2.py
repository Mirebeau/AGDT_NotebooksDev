import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
np.set_printoptions(linewidth=2000)

float_t = ti.f32; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)

class testrecomp_t:
    def __init__(self):self.x = 0
testrecomp = testrecomp_t()

from agdt.Eikonal import Queue2

pq =  Queue2.priority_queue.init(ti.i32,ti.f32,10)

print(pq.size(pq))
print(pq.empty(pq))

@ti.kernel
def testpq(pq_ti:pq.argtype):
    print(pq.size(pq_ti))
    print(testrecomp.x,pq_ti.prio[0])
testpq(pq)

#other_t = ti.types.argpack(a = ti.types.ndarray(), _size=ti.types.ndarray())
other_t = ti.types.argpack(a=pq.argtype,b=pq.argtype)


@ti.kernel
def testpq2(data_ti:other_t):
    print(pq.size(data_ti.a))

#arr = ti.ndarray(ti.f32,10); arr.fill(5)
testpq2(other_t(pq,pq)) #pq,other_t(arr,arr))
pq = pq.with_capacity(pq,20)

testpq(pq)

exit(0)

#pq2 =  Queue2.priority_queue.init(ti.i32,ti.f32,20)
#pq2 = pq.set_capacity(pq,20)
pq2 = Queue2.priority_queue.init(ti.i32,ti.f32,20)

@ti.kernel
def copy_data(old_ti:pq.argtype,val:ti.i32): #,new_ti:pq.argtype):
    for i in range(10):
        old_ti.prio[i] += val #old_ti.prio[i]
copy_data(pq2,3) #,pq2)
#testrecomp.x=1
#testpq(pq2)

print(pq.prio[0])
print(pq2.prio[0])

exit(0)


pq_cls = Queue2.priority_queue
pq_argt,pq_data = Queue2.priority_queue.init(ti.i32,ti.f32)
#print(pq.capacity)

print(pq_cls.size(pq_data))

@ti.kernel
def testpq(data:pq_argt):
    print(pq_cls.size(data))
    #print(pq.size(data))
#    print(testrecomp.x)


testpq(pq_data)

exit(0)

pq = Queue2.priority_queue
pq_type,pq_data = pq.init(ti.i32,ti.f32)
_,pq_data2 = pq.init(ti.i32,ti.f32,capacity=20)

print(pq.capacity(pq_data))

@ti.kernel
def testpq(data:pq_type):
    print(pq_type.capacity(data))
    print(testrecomp.x)
testpq(pq_data)
testrecomp.x = 1
testpq(pq_data2)
