import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
import agdt




from agdt.Eikonal import Metrics
float_t = ti.f32
int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t)
ndim=2
shape = (2,); index=(1,)

if False:
    DecompType = Metrics.Diagonal; traits = DecompType.Traits(ndim,float_t)
    data = traits.vec_t([1,2])
else:
    DecompType = Metrics.Riemann; traits = DecompType.Traits(ndim,float_t)
    data = traits.mat_t([[1,0],[0,1]])

weights = ti.field(float_t, shape = shape+(traits.nactx,))
offsets = ti.Vector.field(ndim,ti.i8, shape = shape+(traits.nactx,))
print(offsets.n)

h = traits.vec_t([2,3])
#print(weights.shape[-1],costs.)
@ti.kernel
def test():
    for _ in range(1):
        DecompType(traits,weights,offsets,index,h,data)

test()

if False:

    if False:
        MetricType = Metrics.diagonal(ndim,float_t)
        metric = MetricType([2.,3.])
    else:
        MetricType = Metrics.riemann(ndim,float_t)
        metric = MetricType([[1.,0.],[0.,1.]])

    v = MetricType.vec_t([4.,5.])

    print(f"{metric.norm2(v)=}")
    print(f"{metric.with_costs(v)}")


    weights = ti.field(float_t,shape=shape+(MetricType.decompdim,)); weights.fill(0)
    offsets = ti.Vector.field(ndim,dtype=ti.i8,shape=shape+(MetricType.decompdim,)); offsets.fill(0)

    metric.decomp(weights,offsets,index)
    print(weights,offsets)

