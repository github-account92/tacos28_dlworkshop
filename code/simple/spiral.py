import numpy as np

from base import plotz


# two spirals
npoints = 100
class1_angle = np.repeat(
    np.linspace(start=0, stop=2*np.pi, num=npoints//2)[None, :],
    2, axis=0).flatten()
class1_radius = np.linspace(start=0, stop=3, num=npoints) + np.random.randn(npoints)/10
c1 = np.array([class1_radius*np.cos(class1_angle), class1_radius*np.sin(class1_angle)]).T

class2_radius = np.linspace(start=0.5, stop=3.5, num=npoints) + np.random.randn(npoints)/10
c2 = np.array([class2_radius*np.cos(class1_angle), class2_radius*np.sin(class1_angle)]).T

points = np.concatenate((c1, c2), axis=0)

plotz(points, "spiral", solve=False)

# in polar coordinates
radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
angles = np.arccos(points[:, 0]/radii)
angles[points[:, 1] < 0] = -angles[points[:, 1] < 0]

plotz(np.stack((radii, angles), axis=1), "spiral_polar", solve=False)
