"""
This file intends to illustrate the fast marching algorithm in the isotropic setting, and reproduce 
some results of the agd implementation.
"""
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import agdt.AD, agdt.GetArrayModule
from agdt.Eikonal import Metrics,HFM
from agd.Plotting import quiver

from matplotlib import pyplot as plt
import taichi as ti
import numpy as np
np.set_printoptions(linewidth=2000)
π = np.pi

float_t = ti.f64; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)

bounds = [[-0.5,0.5],[-0.5,0.5]]
dimx=101
metricType = Metrics.Riemann(2,float_t)
#dom = HFM.Domain(bounds,(5,5),metricType)

dom = HFM.Domain(bounds,(dimx,dimx),metricType)
riemann = ti.field(ti.math.mat2,dom.shape) 

ti_fwd = agdt.AD.fwd_translator()
@ti_fwd
def elevation(x,y): return 0.75*np.sin(3*π*x)*np.sin(3*π*y)

if True: # Build metric using autodiff
    fwd1 = agdt.AD.mk_fwd1(2,float_t)

    @ti.pyfunc
    def topographic_metric(x,y):
        x_ad = fwd1.types.mk(x,0)
        y_ad = fwd1.types.mk(y,1)
        h_ad = elevation(x_ad,y_ad)
        grad = h_ad.v
        return ti.math.eye(2) + grad.outer_product(grad)

    @ti.kernel
    def set_metric():
        for index in ti.grouped(riemann):
            p = dom.PointFromIndex(index)
            riemann[index] = topographic_metric(p[0],p[1])
    set_metric()
elif False: # Build metric using finite differences
    Z = elevation.orig(*dom.grid())
    DxZ,DyZ = np.gradient(Z,dom.h[0],axis=(0,1))
    m = [[1+DxZ**2,DxZ*DyZ],[DxZ*DyZ,1+DyZ**2]]
    riemann.from_numpy( np.moveaxis(m,(0,1),(-2,-1)) )
else: # Constant metric
    riemann = agdt.GetArrayModule.tofield([[1,0],[0,1]],ti.math.mat2)

tips = HFM.Domain(bounds,(6,6),metricType).grid()
tips = np.moveaxis(tips,0,-1).reshape(-1,2)

if True:
    dom.build_scheme(m=riemann)
    dom.set_seed((0,0))
    dom.Algo.solve_FMM()

    print(dom.values(True))
    #exit(0)
    # Note : Some geodesics near the center do not look good (oscillate), but that is expected, since
    # they start close to the cut locus
    ode = dom.ode()
    if True:
        geodesics,codes = ode.backtrack(tips)
        plt.contourf(*dom.grid(),dom.values(True),cmap='Greys')
        for geo in geodesics: plt.plot(*geo.T)
        plt.axis('equal')
        plt.show()

    if False:
        quiver(*dom.grid(),*np.moveaxis(ode.flows.to_numpy(),-1,0),subsampling=(1,1))
        plt.axis('equal')
        plt.show()


# ------ Reproducing the agd results ------
from agd import Eikonal,Metrics
Eikonal.LibraryCall.binary_dir['FileHFM']='/Users/jean-mariemirebeau/bin/FileHFM/Release'
hfmIn = Eikonal.dictIn({
    'model':'Riemann2',
    'seed':[0.,0.],
    'metric':Metrics.Riemann(np.moveaxis(riemann.to_numpy(),(-2,-1),(0,1))),
    'tips':tips,
    'exportValues':True,
    'exportGeodesicFlow':True,
    'geodesicSolver':'ODE'
})
hfmIn.SetRect(sides=bounds,dimx=dimx)
hfmOut = hfmIn.Run()

if False:
    plt.contourf(*hfmIn.Grid(),hfmOut['values'])
    for geo in hfmOut['geodesics']: plt.plot(*geo)
    plt.axis('equal')
    plt.show()

if False:
    quiver(*dom.grid(),*hfmOut['flow'],subsampling=(1,1))
    plt.axis('equal')
    plt.show()

plt.contourf(*dom.grid(),hfmOut['values']-dom.values(True))
plt.axis('equal')
plt.colorbar()
plt.show()

#assert np.allclose(hfmOut['values'],dom.values(True),atol=1e-4)
nbad = np.sum(abs(hfmOut['values']-dom.values(True))>=1e-4)
print(f"Differneces in value function : {nbad=}")
flow_agd = hfmOut['flow']
flow_agdt = np.moveaxis(dom.flows()[0].to_numpy(),-1,0) #np.moveaxis(ode.flows.to_numpy(),-1,0)
nbad = np.sum(np.abs(flow_agd+flow_agdt)>=1e-4)
print(f"Differences in flow {nbad=}")
# Les flow diffèrent seulement en qq points, sur la diagonale. Pas forcément grave, 
# cela correspond au cut-locus.
#print(flow_agd+flow_agdt)
#print(flow_agd)
