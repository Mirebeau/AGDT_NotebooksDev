import sys
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

import numpy as np
import taichi as ti
float_t = ti.f64
ti.init(arch=ti.cpu,default_fp=float_t,debug=True)
np.set_printoptions(linewidth=2000)

from agdt import AD

#fwd_t = AD.mk_fwd1(1,float_t)
fwd_t = AD.fwd0 #mk_fwd0(1,float_t)


#x0_ad = fwd_t(2.,(1.,)
@ti.kernel
def test_fwd():
	x_ad = fwd_t.types.mk(2.,0)

#	x_ad = x0_ad
	for _ in range(1):
#		x_ad.iadd(x_ad)
		y_ad = x_ad.add(x_ad)
		y_ad.print()
		y_ad.iadd(x_ad)
		y_ad.print()
		x_ad.print()
		y_ad.print()

		y_ad.imulc(2.)
		y_ad.print()
		y_ad.isubc(3.)
		y_ad.print()

		z = fwd_t.types.mk(3.,0)
		z.print()

test_fwd()


# Now implent an iteration for some GSD-like problem


