import math
import glob
import numpy as np
from PIL import Image


# parameters

datadir = "./data"
resultdir = "./results"

# you can calibrate these parameters (data dependent)
sigma = 2
highThreshold = 0.03
lowThreshold = 0.01
rhoRes = 2
thetaRes = math.pi / 180
nLines = 20


def replication_padding(Igs, size):
    # TODO
    x, y = size
    x_pad_size = x // 2
    y_pad_size = y // 2
    padded_img = np.pad(
        Igs, ((x_pad_size, x_pad_size), (y_pad_size, y_pad_size)), "edge"
    )
    return padded_img


# Igs : grayscale image
# G: convolution filter
# 이미지 가장자리의 boundary we will pad the value of nearest pixel that lies inside the image
def ConvFilter(Igs, G):
    # TODO ...

    return Iconv  # Igs와 사이즈 동일


def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...

    return Im, Io, Ix, Iy


def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...

    return H


def HoughLines(H, rhoRes, thetaRes, nLines):
    # TODO ...

    return lRho, lTheta


def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...

    return l


def main():

    # read images
    for img_path in glob.glob(datadir + "/*.jpg"):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.0

        # Hough function
        # Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        # H = HoughTransform(Im, rhoRes, thetaRes)
        # lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines)
        # l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == "__main__":
    main()
