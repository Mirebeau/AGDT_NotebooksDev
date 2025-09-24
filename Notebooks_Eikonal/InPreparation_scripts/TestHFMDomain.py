import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
import agdt
np.set_printoptions(linewidth=2000)

from agdt.Eikonal import Metrics,HFM
float_t = ti.f32
int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)
ndim=2

metric = Metrics.Diagonal(ndim,float_t)
#metric.HFMTraits.periodic_axis=0
vec_t = metric.HFMTraits.vec_t
dom = HFM.Domain([[0,1],[0,1]],(10,10),metric)

if(dom.periodic): dom.origin[dom.periodic_axis]+=dom.h[dom.periodic_axis]/2

print(dom.periodic_axis)
print(f"{dom.origin=}")
#print(dom.ih)
#print(dom.sgrid())
#print(dom.shape)

walls = ti.field(metric.HFMTraits.wall_t,shape=dom.shape)
#walls[2,2]=True

dom.build_scheme(walls=walls)
HFM = dom.Algo
x = HFM.Traits.ivec_t(2,2)
ix = HFM.x2ix(x)
#print(f"{HFM.shape}")
#print(f"{ix=}")
walls = HFM.walls.to_numpy().reshape(HFM.shape)
#print("walls",walls)

x = HFM.Traits.ivec_t([1,1])
e = HFM.Traits.ivec_t([1,1])
#print("wall at x+e : ",HFM.walls[HFM.x2ix(x+e)])

#print(HFM.visible(x,e))

#exit(0)

#costs = ti.field(float_t,shape=(1,5))
#for i in range(5): costs[0,i] = i*0.2+0.1

ivec_t2 = ti.math.ivec2
v = vec_t(1.2,3.5)
#print("ivec_t, python scope",ivec_t2(v))

@ti.pyfunc
def printme(ix,iy): 
    print(ix,iy,HFM.ix2x(ix)); print(HFM.ix2x(iy))
#HFM.rneigh(HFM.x2ix(HFM.Traits.ivec_t([1+2,4])),printme)

@ti.kernel
def testInterp():
    #print("ivec_t, taichi scope",ivec_t2(v))
    #print(dom.Interpolate(arr,vec_t(0.2,0.57)))

    norm = metric.NormType([1.,1.])
    #print(norm.norm([3.,4.]))
#    dom.set_seed(vec_t([0.5,0.5]))
    dom.set_seed(vec_t([0.05,0.05]))
    #dom.spread_seed(vec_t([0.7,0.5]),norm,radius=1.5)

testInterp()


#print(dom.values())
print(HFM.walls.to_numpy().reshape(HFM.shape))
#exit(0)
#dom.set_seed(vec_t([0.5,0.5]))

#HFM.set_seed(ix,0)
HFM.solve_FMM()
#HFM.solve_AGSI(1e-4)
#HFM.solve_FastSweeping(1e-4)
#HFM.solve_GlobalIteration(1e-4)

#print(dom.values())
print(HFM.values.to_numpy().reshape(HFM.shape))

x = HFM.Traits.ivec_t([1,1])
#print("adim flow at ",x,HFM.flow(HFM.x2ix(x)))
print("flow at ",x,dom.flow(x))

flows,diffs = dom.flows()
#print(flows)
#print(diffs)
#print(arr[(0,2)])

distL1 = dom.seeds_distL1()
print(distL1)
#v = vec_t([2,4.])
#print(agdt.Linalg.zero_like(v))
#print(v)

# @ti.pyfunc
# def printme(ix,iy): print(ix,iy)
# HFM.rneigh(22,printme)

# shape = (1,1)
# @ti.kernel
# def testloop():
#     for x,y in ti.ndrange(*shape):
#         print(x,y)
# testloop()

ode = dom.ode()
geodesics, geo_code = ode.backtrack([[8,8]])
print(geodesics)
print(geo_code)