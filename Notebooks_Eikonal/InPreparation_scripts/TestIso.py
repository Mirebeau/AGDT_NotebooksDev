"""
This file intends to illustrate the fast marching algorithm in the isotropic setting, and reproduce 
some results of the agd implementation.
"""
import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
from agdt.Eikonal import Metrics,HFM
from agd.Plotting import quiver

from matplotlib import pyplot as plt
import taichi as ti
import numpy as np
np.set_printoptions(linewidth=2000)

float_t = ti.f32; int_t = ti.i32
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)

# Let us first build a domain, the rectangle [-1,1]x[0,1] discretized on a grid, and choose a metric type
dom = HFM.Domain(bounds=[[-1,1],[0,1]], shape=(78,51), metric = Metrics.Diagonal(2,float_t))
X = dom.grid()

# We distinguish between physical points, which range over the rectangular domain, and indices, 
# which range over the grid size
print(f"{dom.IndexFromPoint((-0.5,0.3))=}")
print(f"{dom.PointFromIndex((0.5,0.2))=}")


if False: # ---------------- Solving the eikonal equation in free space --------------
	# We then build the finite difference scheme. Some metric parameters, and walls, could be specified here.
	dom.build_scheme()

	# Set the seds for the front propagation
	dom.set_seed((-0.5,0.3)); dom.set_seed((0.5,0.8),0.8)

	# The eikonal equation can be solved using various numerical methods. Let us choose Fast Marching here
	dom.algo.solve_FMM()
	#dom.algo.solve_AGSI()
	#dom.algo.solve_FastSweeping()
	#dom.algo.solve_GlobalIteration()

	# Once the eikonal equation is solved, we compute the geodesics by solving a backtracking ode
	ode = dom.ode()
	geodesics,codes = ode.backtrack([[0.,0.6],[-0.9,0.5],[0.8,0.8]])
	print(codes)

	if False: # -------- Display the eikonal solution, and the geodesics ---------
		plt.title("Numerical solution of the eikonal equation")
		plt.contourf(*X,dom.values(True))
		plt.axis('equal')
		for geo in geodesics: plt.plot(*geo.T)
		plt.show()

	if False: # --------- Display the geodesic flow vector field ---------
		plt.title("geodesic flow vector field")
		quiver(*X,*np.moveaxis(ode.flows.to_numpy(),-1,0),subsampling=(2,2))
		plt.axis('equal')
		plt.show()

# --------------- Introducing obstacles, and position dependent speed -----------
X,Y = dom.sgrid()
walls = np.logical_or((X-0.3)**2 + (Y-0.3)**2 <= 0.2**2, np.logical_and(X==X[35,0], Y>=0.4))
costs = np.exp(-0.5*(X**2+Y**2))

#def to_field(arr,dtype): field = ti.field(dtype,arr.shape); field.from_numpy(arr); return field

def to_ndarray(arr,dtype): a = ti.ndarray(dtype,arr.shape); a.from_numpy(arr); return a


if False: # Run with obstacles and a cost function
	dom.build_scheme(to_field(costs,float_t),to_field(walls,ti.i8))
	dom.set_seed((-0.5,0.3)); dom.set_seed((0.5,0.8),0.5)

	dom.algo.solve_FMM()
	geodesics,codes = dom.ode().backtrack([[0.,0.6],[-0.9,0.5],[0.8,0.8]])

	plt.title("Numerical solution of the eikonal equation")
	plt.contourf(*dom.grid(),dom.values(True),cmap='Greys')
	plt.axis('equal')
	for geo in geodesics: plt.plot(*geo.T)
	plt.show()


if True: # Custom stopping criterion
	@ti.pyfunc
	def stopping_criterion(algo_ti:ti.template(),ix):
		# Here we stop the FMM at a prescribed distance.
		# Alternatively, stop when all tips are accepted, etc
		return algo_ti.values[ix]>=0.7 

#	dom.build_scheme(to_n)
#	dom.build_scheme(to_field(costs,float_t),to_field(walls,ti.i8))
	dom.build_scheme(to_ndarray(costs,float_t),to_ndarray(walls,ti.i8))
	dom.set_seed(dom.self_ti,(-0.5,0.3)); dom.set_seed(dom.self_ti,(0.5,0.8),0.5)

	#dom.algo.solve_FMM(stopping_criterion)
	#dom.algo.solve_AGSI(1e-4)
	dom.algo.solve_FastSweeping(3e-4)
	#dom.algo.solve_GlobalIteration(1e-4)

	geodesics,codes = dom.ode().backtrack([[0.,0.6],[-0.9,0.5],[0.8,0.8]])
#	geodesics,codes = dom.ode().backtrack([[0.,0.6],[-0.9,0.5],[0.8,0.8], [0.8,0.9]])
	plt.title("Numerical solution of the eikonal equation")
	plt.contourf(*dom.grid(),dom.values(True),cmap='Greys')
	plt.axis('equal')
	for geo in geodesics: plt.plot(*geo.T)
	plt.show()
