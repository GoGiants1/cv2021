import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils


def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """
    list_of_gaussian_pyramid = [np.array(input_image, "f")]
    for _ in range(level):
        level_img = utils.down_sampling(list_of_gaussian_pyramid[-1])
        list_of_gaussian_pyramid.append(level_img)
    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    return list_of_gaussian_pyramid


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    list_of_laplacian = []
    gaussian_len = len(gaussian_pyramid)
    for i, gaussian_img in enumerate(gaussian_pyramid):
        if i + 1 <= gaussian_len - 1:
            expanded_img = utils.up_sampling(gaussian_pyramid[i + 1])
            list_of_laplacian.append(utils.safe_subtract(gaussian_img, expanded_img))
    list_of_laplacian.append(gaussian_pyramid[-1])

    return list_of_laplacian


def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    # mask는 image 2와 , 1 - mask는 image 1과
    image1_sub_mask = utils.safe_subtract(image1, mask)
    img1_gp = gaussian_pyramid(image1_sub_mask, level)
    img2_gp = gaussian_pyramid(image2, level)
    img1_lp = laplacian_pyramid(img1_gp)
    img2_lp = laplacian_pyramid(img2_gp)
    merged_lps = []
    # blended_img
    for i in range(level, -1, -1):
        merged_lps.append(utils.safe_add(img1_lp[i], img2_lp[i]))

    prev_merged_image = merged_lps[0]
    for i, _ in enumerate(merged_lps):
        if i + 1 <= level:
            up_sampled_lp = utils.up_sampling(prev_merged_image)
            # print(f"upsampled: {up_sampled_lp.shape}")
            prev_merged_image = utils.safe_add(merged_lps[i + 1], up_sampled_lp)
    return prev_merged_image


if __name__ == "__main__":
    hand = np.asarray(Image.open(os.path.join("images", "hand.jpeg")).convert("RGB"))
    flame = np.asarray(Image.open(os.path.join("images", "flame.jpeg")).convert("RGB"))
    mask = np.asarray(Image.open(os.path.join("images", "mask.jpeg")).convert("RGB"))

    logdir = os.path.join("results", "HW1_2")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3

    plt.figure()
    plt.imshow(Image.open(os.path.join("images", "direct_concat.jpeg")))
    plt.axis("off")
    plt.savefig(os.path.join(logdir, "direct.jpeg"))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis("off")
        plt.savefig(os.path.join(logdir, "gaussian_pyramid.jpeg"))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis("off")
            plt.savefig(os.path.join(logdir, "laplacian_pyramid.jpeg"))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis("off")
        plt.savefig(os.path.join(logdir, "blended.jpeg"))
        plt.show()
