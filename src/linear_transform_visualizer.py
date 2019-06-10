import os
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio as im


def colorizer(x, y):
    '''
    Default colorizer for LinearTransformVizualizer

    '''
    r = min(1, 1-y/3)
    g = min(1, 1+y/3)
    b = 1/4 + x/16
    return (r, g, b)


class LinearTransformVizualizer:

    '''
    LinearTransformVizualizer makes it able to visualize 2-D and 3-D
    linear transforms given with 2x2 and 3x3 matrix.
        transform -- 2x2 or 3x3 array representing transform matrix;
                     successive arrays represent columns of the matrix
        colorizer -- function used to get unique color basing on the
                     x,y,z coordinates. Returns (r, g, b) rouple where
                     r,g,b are placed in a range (0; 1)
        steps -- number of images constituting gif animation;
                 default 30
        data -- 2-D array in size of (2 x n) - where n is number of
                samples - containing 2 arrays of succesive x/y 
                coordinates
    '''

    def __init__(self, transform, colorizer=colorizer, data=[], steps=30):
        self.transform = transform
        self.colorizer = colorizer
        self.data = data
        self.steps = steps

    def _stepwise_transform(self):
        '''
        Generate a series of intermediate transforms for self.transform
        matrix and self.data points; matrix multiplication starts
        with the identity matrix.

        Returns a (self.steps + 1)-by-2-by-(transform_rank) array.

        '''
        # create empty array of the intermediate data
        transgrid = np.zeros((self.steps + 1,) + np.shape(self.data))

        # compute intermediate transforms
        for j in range(self.steps + 1):
            intermediate = np.eye(2) + j/self.steps * \
                (self.transform - np.eye(2))
            # apply intermediate matrix transformation
            transgrid[j] = np.dot(intermediate, self.data)

        return transgrid

    def visualize(self, gifname, figuresize=(4, 4), dpi=150):
        '''
        Generates a series of png images showing a linear transformation stepwise
        and creates .gif animation of these images.

        '''

        # getting intermediate transforms
        transarray = self._stepwise_transform()

        # creating set of colors for points in the scatter
        color = list(self.colorizer(x, y)
                     for x, y in zip(self.data[0], self.data[1]))

        # to determine filename padding
        ndigits = len(str(transarray.shape[0]))
        maxval = np.abs(transarray.max())  # to set axis limits

        # create directory if necessary
        outdir = "tmp"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # create figure
        plt.ioff()
        fig = plt.figure(figsize=figuresize)

        # files' names for further gif creating
        names = []

        # plot individual frames
        for j in range(transarray.shape[0]):
            plt.cla()
            plt.scatter(transarray[j, 0], transarray[j, 1],
                        s=36, c=color, edgecolor="none")
            plt.xlim(1.1*np.array([-maxval, maxval]))
            plt.ylim(1.1*np.array([-maxval, maxval]))
            plt.grid(True)
            plt.draw()
            # save as png
            name = "frame-" + str(j+1).zfill(ndigits) + ".png"
            names.append(outdir + "\\" + name)
            outfile = os.path.join(
                outdir, name)
            fig.savefig(outfile, dpi=dpi, facecolor="c")

        # create gif
        images = []

        for filename in names:
            images.append(im.imread(filename))

        im.mimsave(gifname, images)

        # deleting tmp folder and images
        shutil.rmtree(outdir)

        # tuning-on interactive mode
        plt.ion()
