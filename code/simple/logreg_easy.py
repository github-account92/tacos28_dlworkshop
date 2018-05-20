import numpy as np

from base import plotz


# two easily separable point clouds
npoints = 100
np.random.seed(5)
c1 = np.random.randn(npoints, 2) - [[1.5, 0.3]]
c2 = np.random.randn(npoints, 2) + [[1.7, 0.8]]
points = np.concatenate((c1, c2), axis=0)

plotz(points, "easy", solve=False, eb=0.6)
plotz(points, "easy_solved", eb=0.6)
