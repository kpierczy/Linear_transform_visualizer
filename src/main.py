import numpy as np
import linear_transform_visualizer as vs

# transform's matrix
A = np.column_stack([[1, 0], [1, 1]])


if __name__ == '__main__':
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-3, 3, 7)
    xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])

    visualizer = vs.LinearTransformVizualizer(A, data=xygrid)
    visualizer.visualize("mytransform.gif")
