import sys; sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.append("/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
from agdt.Eikonal import NarrowBand

import numpy as np
import taichi as ti
import taichi.algorithms._algorithms as ti_algo

float_t = ti.f32; int_t = ti.i32; arr_t = ti.types.ndarray()
ti.init(arch=ti.gpu,default_fp=float_t,default_ip=int_t, debug=True)
np.set_printoptions(linewidth=2000)

if False: # Test prefix sum (unsupported on metal)
	n = 100
	arr = ti.ndarray(ti.i32,n); arr.fill(1)
	prefix_sum_exec = ti_algo.PrefixSumExecutor(n)
	prefix_sum_exec.run(arr)
	print(arr.to_numpy())
	exit(0)

if False: # Test fast-sweeping ordering
	shape = (2,3,4)
	ixs = NarrowBand._Algo.enumerate_sweeps(shape).to_numpy()
	for k in range(len(shape)):
		size = np.prod(shape)
		ksize = size//shape[k]
		for r in range(shape[k]):
			beg = k*size + r*ksize
			print(ixs[beg:beg+ksize])

shape = (6,6)
bounds = [[0,1],[0,1]]
#metric = NarrowBand.Laplacian(2,float_t)
metric = NarrowBand.DistL1(2,float_t)
Traits = metric.NBTraits
Traits.shape_i = (4,4)
Traits.niter_i = 2
Traits.strict_iter_o = True
#metric.NBTraits.niter_i = 0
dom = NarrowBand.Domain(bounds,shape,metric)
dom.build_scheme()
algo = dom.algo
print(metric.NBTraits.stencil)

#periodic = (False,False)
#shape_i = (2,2)
#shape_o = (2,2)
#print(Traits.cprod_i)
#algo = NarrowBand._Algo(metric,shape)

#ixs_o = ti.ndarray(ti.i32,2); ixs_o.from_numpy(np.array([0,1]))
#print(NarrowBand.axis_aligned_stencil(2))
#improved = ti.ndarray(ti.i8,ixs_o.shape[0]); improved.fill(False)

#algo.update(dom.algo.self_ti, ixs_o, improved)
#algo.update(dom.algo.self_ti, ixs_o, improved)
#algo.values[algo.x2ix(Traits.ivec_t(1,1))]=0
#print(algo.values.to_numpy())
# @ti.kernel
# def set_values_range(values:arr_t):
# 	for x in ti.grouped(ti.ndrange(*(Traits.shape_i*algo.shape_o))):
# 		values[algo.x2ix(x)] = x[0]
# set_values_range(algo.values)           

seed = Traits.ivec_t(2,2)
dom.set_seed(algo.self_ti,seed)

if False:
	size = algo.size_o
	ixs = ti.ndarray(ti.i32,size); ixs.from_numpy(np.array([0,2,1,0]))
	improved = ti.ndarray(ti.i8,size); improved.from_numpy(np.array([1,1,1,0]))
	ixs_new = ti.ndarray(ti.i32,size); ixs_new.fill(-1)
	tag = ti.ndarray(ti.i32,size); tag.fill(-1)
	tag_count = ti.ndarray(ti.i32,size); tag_count.fill(0)
	ixs_end=3
	ixs_end_new = algo.tag_neighbors(ixs,improved,3,ixs_new,tag,tag_count)

	print('tag',tag.to_numpy().reshape(algo.shape_o))
	print('tag_count',tag_count.to_numpy())
	print('ixs_new',ixs_new.to_numpy())
	print(ixs_end_new)
	exit(0)

iter = algo.solve_AGSI(0,10)
print(f"{iter=}")
#algo.solve_FastSweeping(0,1)
#algo.solve_GlobalIteration(0,2)
#print(algo.values.to_numpy())
print(algo.block_squeeze(algo.new_values,remove_pad=True).to_numpy())


#flow_oi = algo.flows()
#flow = algo.block_squeeze(flow_oi,remove_pad=True)
print(dom.flows().to_numpy())
