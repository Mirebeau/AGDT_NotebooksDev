# We check that the position-momentum, and the velocity-stress implementations match, in 
# dimension one and two

import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
import taichi as ti

float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)
np.set_printoptions(linewidth=2000)

import agdt
#from agdt import Selling
from agdt.Waves.AnisoScalar import AnisoScalar,BoxNormal
np_float_t = agdt.convert_dtype['np'][float_t]

#from agd.Metrics.misc import expand_symmetric_matrix
#from matplotlib import pyplot as plt

dt=1; dx=1
shape = (5,5)

E = np.eye(2).astype(int)
decompdim = E.shape[1]

np.random.seed(42)
μ = 1+np.random.rand(*shape) # Inverse density
λ = 1+np.random.rand(*shape,decompdim) # Decomposition coefficients
#μ = np.ones(shape,dtype=np_float_t)
#λ = np.ones((*shape,decompdim),dtype=np_float_t) # Isotropic Laplacian
wave = AnisoScalar(μ,λ,E,dx,dt)

print("Γσ\n",wave.Γσ)

assert np.allclose(wave.Γv.to_numpy(),1)

q = ti.field(float_t,wave.size); q.from_numpy(np.random.rand(*q.shape)-0.5)
p = ti.field(float_t,wave.size); p.from_numpy(np.random.rand(*p.shape)-0.5)
σ = wave.q2σ(q)
v = wave.p2v(p)

H_orig = wave.Hqp(q,p,'orig')
assert np.allclose(H_orig,wave.Hσv(σ,v,'orig'))

Hp = wave.Hqp(q,p)
assert np.allclose(Hp,wave.Hσv(σ,v))

p1 = ti.field(float_t,wave.size); p1.copy_from(p)
q1 = ti.field(float_t,wave.size); q1.copy_from(q)
wave.Verlet_p(q1,p1)
assert np.allclose(Hp,wave.Hqp(q1,p1))

σ1 = wave.q2σ(q)
v1 = wave.p2v(p)
wave.Verlet_v(σ1,v1)
assert np.allclose(Hp,wave.Hσv(σ1,v1))
assert np.allclose(σ1.to_numpy(),wave.q2σ(q1).to_numpy())
assert np.allclose(v1.to_numpy(),wave.p2v(p1).to_numpy())



