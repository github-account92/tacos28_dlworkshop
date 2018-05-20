from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt


def plotz(points, fname, solve=True, eb=1):
    per_class = points.shape[0] // 2
    classes = np.concatenate((np.zeros(per_class), np.ones(per_class)), axis=0)

    plt.scatter(points[:per_class, 0], points[:per_class, 1],
                c="#ff5154")
    plt.scatter(points[per_class:, 0], points[per_class:, 1],
                c="#91a6ff", marker="x")

    xleft, xright = np.min(points[:, 0]) - eb, np.max(points[:, 0]) + eb
    boundb, boundu = np.min(points[:, 1]) - eb, np.max(points[:, 1]) + eb

    if solve:
        solver = LogisticRegression(C=100)
        solver.fit(points, classes)
        c1, c2 = solver.coef_[0]
        b = solver.intercept_

        yleft = ((-c1 * xleft - b) / c2)[0]
        yright = ((-c1 * xright - b) / c2)[0]

        plt.plot([xleft, xright], [yleft, yright], "k-", lw=2)
        print("COEFFICIENTS: C1 {} C2 {} B {}".format(c1, c2, b))

    plt.ylim((boundb, boundu))
    plt.xlim((xleft, xright))

    cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_visible(False)
    #cur_axes.axes.get_yaxis().set_visible(False)

    plt.savefig(fname)
    plt.show()

    if solve:
        return xleft, xright, yleft, yright