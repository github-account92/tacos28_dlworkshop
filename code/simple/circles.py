import numpy as np

from base import plotz


# two easily separable circles
npoints = 100
np.random.seed(5)
c1 = np.random.randn(npoints, 2)/5
# generate random polar coordinates and then transform
class2_radius = np.random.uniform(low=0.7, high=1, size=npoints)
class2_angle = np.random.uniform(low=0, high=2*np.pi, size=npoints)
c2 = np.array([class2_radius*np.cos(class2_angle),
               class2_radius*np.sin(class2_angle)]).T
points = np.concatenate((c1, c2), axis=0)

plotz(points, "circles",  solve=False, eb=0.2)

# in polar coordinates
radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
angles = np.arccos(points[:, 0]/radii)
angles[points[:, 1] < 0] = -angles[points[:, 1] < 0]

xl, xr, yl, yr = plotz(np.stack((radii, angles), axis=1), "circles_polar",
                       eb=0.2)

# some CRAZY stuff: projecting polar solution back into cartesian space
x_interp = np.linspace(xl, xr, num=1000)
y_interp = np.interp(x_interp, xp=np.array([xl, xr]), fp=np.array([yl, yr]))

x_subs = x_interp[np.logical_and(y_interp > -np.pi, y_interp < np.pi)]
y_subs = y_interp[np.logical_and(y_interp > -np.pi, y_interp < np.pi)]

wut = np.array([x_subs*np.cos(y_subs), x_subs*np.sin(y_subs)]).T
from matplotlib import pyplot as plt
plt.scatter(wut[:, 0], wut[:, 1])
plt.show()


