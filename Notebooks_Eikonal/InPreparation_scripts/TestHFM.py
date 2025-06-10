import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
import agdt
from agdt import Sort

from agdt.Eikonal import HFM
float_t = ti.f32
int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t)

shape = (7,7)
ndim = len(shape)
costs = ti.field(dtype=float_t,shape=shape); costs.fill(1)
weights = ti.field(dtype=float_t,shape=shape+(ndim,)); weights.fill(1)
offsets = ti.Vector.field(ndim,dtype=ti.i8,shape=shape+(ndim,)); offsets.fill(0)
@ti.kernel # Set offsets to identity
def set_offsets():
    for I in ti.grouped(ti.ndrange(*shape)):
        for i in ti.static(range(ndim)): offsets[*I,i][i] = 1
set_offsets()

hfm = HFM.HFM(costs,weights,offsets)

x = hfm.ivec_t((3,3))
e = hfm.ivec_t((-2,2))
ix = hfm.x2ix(x)
print(f"{x=}, {ix=}, {hfm.ix2x(ix)=}")
print(f"{hfm.shape=}, {hfm.size=}, {hfm.ndim=}")
print(f"{hfm.factored=}, {hfm.factored_start=}, {hfm.factored_stop=}, {hfm.factored_div=}, {hfm.factored_size=}")
print(f"{hfm.indomain(x)=}, {hfm.visible(x,e)=}, {hfm.visible(x,2*e)=}")
print(f"{hfm.factored_index(ix)=}")

# Reversed stencil test
@ti.pyfunc
def print_neigh(ix,iy): print(iy)
# @ti.pyfunc
# def callme(x,f:ti.template()): f(x)
hfm.rneigh(ix,print_neigh)
hfm.values[ix] = 0
@ti.kernel
def test_roffsets():
    for _ in range(1):
        hfm.rneigh(ix, print_neigh)
        print(f"{Sort.argsort(x)=}")
        hfm.update(ix+1)
test_roffsets()
print(f"{Sort.argsort(x)=}")
print(hfm.values.to_numpy().reshape(hfm.shape))

print(hfm.walls[ix])
hfm.update(ix-1) # Fails with -1
print(hfm.values.to_numpy().reshape(hfm.shape))

if True: # FMM
    hfm.values.fill(np.inf); hfm.walls.fill(0)
    hfm.set_seed(ix,0)
    hfm.solve_FMM()
    print(hfm.values.to_numpy().reshape(hfm.shape))

if False: # FMM with stopping criterion
    @ti.pyfunc
    def stopping_criterion(iy):
        return iy==ix+2 # Stop when point of index ix+2 is reached
    hfm.values.fill(np.inf); hfm.walls.fill(0)
    hfm.set_seed(ix,0)
    hfm.solve_FMM(stopping_criterion)
    print(hfm.values.to_numpy().reshape(hfm.shape))


if False:
    hfm.values.fill(np.inf); hfm.walls.fill(0)
    hfm.set_seed(ix,0)
    hfm.solve_AGSI(1e-6)
    print(hfm.values.to_numpy().reshape(hfm.shape))

if False:
    hfm.values.fill(np.inf); hfm.walls.fill(0)
    hfm.set_seed(ix,0)
    hfm.solve_FastSweeping(1e-6)
    print(hfm.values.to_numpy().reshape(hfm.shape))

if False:
    hfm.values.fill(np.inf); hfm.walls.fill(0)
    hfm.set_seed(ix,0)
    hfm.solve_GlobalIteration(1e-6)
    print(hfm.values.to_numpy().reshape(hfm.shape))

grad = hfm.gradient(ix+1+7)
print(grad)