"""
This file intends to illustrate the fast marching algorithm in the isotropic setting, and reproduce 
some results of the agd implementation.
"""

import numpy as np
import taichi as ti
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
import agdt
from agdt.Eikonal import Metrics,HFM
np.set_printoptions(linewidth=2000)

float_t = ti.f32; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)

# Let us first build a domain, the rectangle [-1,1]x[0,1] discretized on a grid, and choose a metric type
dom = HFM.Domain(bounds=[[-1,1],[0,1]], shape=(100,50), metric = Metrics.Diagonal(2,float_t))

# We distinguish between physical points, which range over the rectangular domain, and indices, 
# which range over the grid size
print(f"{dom.IndexFromPoint((-0.5,0.3))=}")
print(f"{dom.PointFromIndex((0.5,0.2))=}")

# We then build the finite difference scheme. Some metric parameters, and walls, could be specified here.
dom.build_scheme()

# Set the seds for the front propagation
dom.set_seed((-0.5,0.3))
dom.set_seed((0.5,0.8),0.8)

# The eikonal equation can be solved using various numerical methods. Let us choose Fast Marching here
dom.Algo.solve_FMM()
