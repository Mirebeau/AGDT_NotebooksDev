# AGDT_NotebooksDev
Notebooks being written for the AGDT library (stored with full output)

This working repository contains notebooks, many of them in development, meant to illustrate the AGDT library.
The notebooks are stored with their full outputs, which makes diffs unusable and the repo heavy.

Please consider the AdaptiveGridDiscretizations_Taichi repository, where the notebooks are stored without ouputs, hence with usable diffs.

**Installation.**
`pip install agdt'
Alternatively, simply download the library, which is pure python, and add the containing directory to the path:
import sys; sys.path.insert(0,'directory containing agdt')

**Environnement.**
We use Taichi since it is:
- portable (cpu,gpu,metal,cuda)
- transparent (explicit for loops, fields)

Note that taichi requires python 3.11 (as of dec 2024). For a portable fft (cpu,cuda,metal), we use pytorch. 

`pip install taichi`
`conda install pytorch torchvision -c pytorch`

**Arrays**
Taichi fields and numpy ndarrays need to be converted to each other.
Taichi fields allow specifying nice element types. Generic programming with `ti.template()'
Numpy arrays allow using external libraries, in particular FFT, and linear solves. Generic programming with `ti.types.ndarray()'
