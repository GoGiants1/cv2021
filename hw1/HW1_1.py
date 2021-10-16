import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math


def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array) -> 각각의 변에서 2개의 픽셀씩 패딩(0값) 생겨야함.
    """

    def check_nearest_vertex(vertices, coordinate):
        curr_x, curr_y = coordinate
        min_len = math.inf
        min_x, min_y = -1, -1
        for x, y in vertices:
            dist = (curr_x - x) ** 2 + (curr_y - y) ** 2
            if min_len > dist:
                min_len = dist
                min_x, min_y = x, y
        return min_x, min_y

    def add_padding(img_2d, pad_size):
        org_width, org_height = img_2d.shape
        x_pad, y_pad = pad_size
        new_img_width, new_img_height = org_width + 2 * x_pad, org_height + 2 * y_pad
        new_img_2d = np.zeros((new_img_width, new_img_height), np.uint8)

        # 새로운 행렬에서 원래 행렬의 좌표 경계점(배열의 인덱스)
        origin_in_new_r_start = y_pad
        origin_in_new_c_start = x_pad
        origin_in_new_r_end = new_img_height - y_pad - 1
        origin_in_new_c_end = new_img_width - x_pad - 1

        new_img_2d[
            origin_in_new_r_start : origin_in_new_r_end + 1,
            origin_in_new_c_start : origin_in_new_c_end + 1,
        ] = img_2d
        vertices = [
            (x, y)
            for x in (origin_in_new_r_start, origin_in_new_r_end)
            for y in (origin_in_new_c_start, origin_in_new_c_end)
        ]
        # 선대칭 4번, 점 대칭 2번
        padding_rows = [
            y
            for y in [
                *range(origin_in_new_r_start),
                *range(origin_in_new_r_end + 1, new_img_height),
            ]
        ]
        padding_cols = [
            x
            for x in [
                *range(origin_in_new_c_start),
                *range(origin_in_new_c_end + 1, new_img_width),
            ]
        ]
        for new_r_idx, _ in enumerate(new_img_2d):
            if new_r_idx in padding_rows:
                # 선대칭 기준이 되는 row
                ref_row = (
                    origin_in_new_r_start
                    if new_r_idx < origin_in_new_r_start
                    else origin_in_new_r_end
                )
                new_img_2d[
                    new_r_idx, origin_in_new_c_start : origin_in_new_c_end + 1
                ] = new_img_2d[
                    2 * ref_row - new_r_idx,
                    origin_in_new_c_start : origin_in_new_c_end + 1,
                ]

                for new_c_idx in padding_cols:
                    near_x, near_y = check_nearest_vertex(
                        vertices, (new_r_idx, new_c_idx)
                    )
                    new_img_2d[new_r_idx, new_c_idx] = new_img_2d[
                        2 * near_x - new_r_idx,
                        2 * near_y - new_c_idx,
                    ]
            else:
                for new_c_idx in padding_cols:
                    ref_col = (
                        origin_in_new_c_start
                        if new_c_idx < origin_in_new_c_start
                        else origin_in_new_c_end
                    )
                    new_img_2d[new_r_idx, new_c_idx] = new_img_2d[
                        new_r_idx,
                        2 * ref_col - new_c_idx,
                    ]

        return new_img_2d

    # image matrix size (x, y, z)
    img_x_size, img_y_size, img_channels = input_image.shape
    # kernel size and padding size
    x, y = size
    x_pad_size = x // 2
    y_pad_size = y // 2

    # RGB channels
    r = input_image[:, :, 0]
    g = input_image[:, :, 1]
    b = input_image[:, :, 2]
    rgb_list = [r, g, b]

    # new numpy array
    new_img = np.zeros(
        (img_x_size + 2 * x_pad_size, img_y_size + 2 * y_pad_size, 3), np.uint8
    )

    for i in range(img_channels):
        new_img[:, :, i] = add_padding(rgb_list[i], (x_pad_size, y_pad_size))

    return new_img


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array == 패딩 없는 이미지
        Kernel (numpy array): kernel shape of (height, width)  -> 실제 커널 행렬
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image

    reflected_img = reflect_padding(input_image, Kernel.shape)

    # padded img size
    # pad_img_width, pad_img_height, pad_img_channels = reflected_img.shape
    # kernel size and padding size
    x_kernel, y_kernel = Kernel.shape
    x_pad_size = x_kernel // 2
    y_pad_size = y_kernel // 2

    # original image matrix size (x, y, z)
    origin_width, origin_height, origin_channels = input_image.shape
    out_matrix = np.zeros((origin_width, origin_height, origin_channels))

    fliped_kernel = np.flip(Kernel)
    for i in range(origin_channels):
        channel = reflected_img[:, :, i]
        # 원래 이미지 기준에서의 픽셀 좌표
        origin_x, origin_y = 0, 0
        tmp_out = []
        while origin_x < origin_height:
            reflected_x, reflected_y = origin_x + x_pad_size, origin_y + y_pad_size
            pix_result = []
            for r in range(-x_pad_size, x_pad_size + 1):
                for c in range(-y_pad_size, y_pad_size + 1):
                    pix_result.append(
                        channel[reflected_x + r, reflected_y + c]
                        * fliped_kernel[r + x_pad_size, c + y_pad_size]
                    )
            tmp_out.append(sum(pix_result))
            origin_y += 1
            if origin_y >= origin_width:
                origin_y = 0
                origin_x += 1
            if origin_x >= origin_height:
                out_matrix[:, :, i] = np.array(tmp_out).reshape(
                    origin_width, origin_height
                )
                break

    return out_matrix


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
    origin_width, origin_height, origin_channels = input_image.shape

    padded_img = reflect_padding(input_image, size)
    output_img = np.zeros((origin_width, origin_height, origin_channels))
    x_filter_size, y_filter_size = size
    x_pad_size, y_pad_size = x_filter_size // 2, y_filter_size // 2

    for i in range(origin_channels):
        channel = padded_img[:, :, i]

        origin_x, origin_y = 0, 0
        tmp_out = []
        while origin_x < origin_width:
            reflected_x, reflected_y = origin_x + x_pad_size, origin_y + y_pad_size
            pixel_vec = []
            for r in range(-x_pad_size, x_pad_size + 1):
                pixel_vec = [
                    *pixel_vec,
                    *channel[
                        reflected_x + r,
                        reflected_y - y_pad_size : reflected_y + y_pad_size + 1,
                    ],
                ]
            tmp_out.append(sorted(pixel_vec)[x_filter_size * y_filter_size // 2 + 1])
            # tmp_out.append(statistics.median(pixel_vec))
            origin_y += 1
            if origin_y >= origin_width:
                origin_y = 0
                origin_x += 1
            if origin_x >= origin_height:
                output_img[:, :, i] = np.array(tmp_out).reshape(
                    origin_width, origin_height
                )
                break

    return output_img


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

    def gaussian_kernel_1d(sig, f_size):
        domain = range(-(f_size // 2), f_size // 2 + 1)
        gaussian_kernel = [
            (np.exp(-((x ** 2) / (2.0 * sig ** 2))) / (sig * math.sqrt(2.0 * np.pi)))
            for x in domain
        ]
        total_sum = np.sum(gaussian_kernel)
        return gaussian_kernel / total_sum

    # Your code
    # padded_img = reflect_padding(input_image, size)
    origin_width, origin_height, origin_channels = input_image.shape
    x_filter_size, y_filter_size = size
    x_kernel = np.array(gaussian_kernel_1d(sigmax, x_filter_size)).reshape(
        x_filter_size, 1
    )
    y_kernel = np.array(gaussian_kernel_1d(sigmay, y_filter_size)).reshape(
        1, y_filter_size
    )

    gaussian_kernel_2D = np.dot(x_kernel, y_kernel)

    result = convolve(input_image, gaussian_kernel_2D)
    # result = convolve(result, y_kernel)
    return result


if __name__ == "__main__":
    image = np.asarray(
        Image.open(os.path.join( "images", "baboon.jpeg")).convert("RGB")
    )
    # image = np.asarray(
    #     Image.open(os.path.join("images", "gaussian_noise.jpeg")).convert("RGB")
    # )
    # # image = np.asarray(
    # #     Image.open(
    # #         os.path.join( "images", "salt_and_pepper_noise.jpeg")
    # #     ).convert("RGB")
    # # )

    logdir = os.path.join("results", "HW1_1")
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
