"""
This file intends to illustrate the fast marching algorithm with non-holonomic metrics, and reproduce 
some results of the agd implementation.
"""

import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
from agdt.GetArrayModule import to_ndarray
from agdt.Eikonal import Metrics,HFM
from agd.Plotting import quiver

from matplotlib import pyplot as plt
import taichi as ti
import numpy as np
np.set_printoptions(linewidth=2000)
π = np.pi

float_t = ti.f64; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=False)

# for a,b in [(-5,3),(5,3)]:
#     idiv = HFM.div_round_closest(a,b)
#     print(idiv,np.round(a/b))

# print(HFM.div_round_closest(-4*4,4))
# print((-16-2)//4)
# print(-1//5)
# exit(0)

# ---------- Define the model ---------

bounds = [[-1,1],[0,1],[0,2*π]]
dims = (53,25,64)
#dims = (100,50,32)
#metric = Metrics.ReedsShepp2(float_t)
#metric = Metrics.ReedsSheppForward2(float_t)
#metric = Metrics.Dubins2(float_t) # FAILS (vanishing flow)
metric = Metrics.Elastica2(float_t) # FAILS compilation. TODO : fix merge sort


dom = HFM.Domain(bounds,dims,metric)
X,Y,θ = dom.grid() # Dense grid of the domain (numpy arrays)

# ---------- Optional : walls --------
disk = (X-0.3)**2 + (Y-0.3)**2 <= 0.2**2
barrier = np.logical_and(X==X[dims[0]//3,0,0], Y>=0.4)
walls = np.logical_or(disk,barrier) 

# ---------- Optional : cost ----------
costs = 1+3*(X<=0)
print(f"{costs.shape=},{walls.shape=}")

dom.build_scheme(to_ndarray(costs,float_t),to_ndarray(walls,ti.i8))
#dom.build_scheme(to_ndarray(costs,float_t)) 
#dom.build_scheme(walls=to_ndarray(walls,ti.i8))
#dom.build_scheme() 

seed = dom.Traits.vec_t((-0.5,0.5,π/2))
#_,_,θ = dom.sgrid()
dom.set_seed(dom.self_ti, seed)
dom.algo.solve_FMM()
#dom.algo.solve_AGSI(1e-4)
#dom.algo.solve_FastSweeping(1e-4)
#dom.algo.solve_GlobalIteration(1e-4)

ode = dom.ode()
tips = [[0.5,0.5,π/2]]
geodesics,rcodes = ode.backtrack(tips,delay_values=40)
#print(geodesics)
print(rcodes)
#exit(0)

plt.contourf(X[:,:,0],Y[:,:,0],costs[:,:,0]+10*walls[:,:,0],cmap='Greys')
plt.plot(*geodesics[0].T[:2])
plt.axis('equal')
plt.show()