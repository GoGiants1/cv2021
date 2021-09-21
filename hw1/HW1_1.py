import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array) -> 각각의 변에서 2개의 픽셀씩 패딩(0값) 생겨야함.
    """
    # image matrix size (x, y, z)
    img_x_size, img_y_size, _ = image.shape
    # kernel size and padding size
    x, y = size
    x_pad_size = x // 2
    y_pad_size = y // 2
    # RGB channels
    r = input_image[:, :, 0]
    g = input_image[:, :, 1]
    b = input_image[:, :, 2]

    # add padding each channels
    r = np.pad(r, ((x_pad_size, x_pad_size), (y_pad_size, y_pad_size)), "reflect")
    g = np.pad(g, ((x_pad_size, x_pad_size), (y_pad_size, y_pad_size)), "reflect")
    b = np.pad(b, ((x_pad_size, x_pad_size), (y_pad_size, y_pad_size)), "reflect")

    # new numpy array
    new_img = np.zeros(
        (img_x_size + 2 * x_pad_size, img_y_size + 2 * y_pad_size, 3), np.uint8
    )

    new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = r, g, b

    return new_img


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    return


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")

    # Your code
    return


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    return


if __name__ == "__main__":
    image = np.asarray(
        Image.open(os.path.join(os.getcwd(), "images", "baboon.jpeg")).convert("RGB")
    )
    # image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    # image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join(os.getcwd(), "results", "HW1_1")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5, 5)) / 25.0
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)

    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "reflect.jpeg"))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "convolve.jpeg"))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "median.jpeg"))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "gaussian.jpeg"))
        plt.show()
