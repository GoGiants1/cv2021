import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold


def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    """lucas_Kanade_affine
    dp = delta p = H^
    img1: I(t): template -> img2: I(t+1) : warped image
    Gx = I_x
    Gy = I_y
    """
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    max_gradient = max(Gx.max(), Gy.max())

    # scaling to 0 ~ 1
    G = np.hstack((Gx.reshape((-1, 1)), Gy.reshape((-1, 1)))) / max_gradient
    # r,c => y,x size
    img1_r, img1_c = img1.shape
    img2_r, img2_c = img2.shape

    # for homogeneous coordinate calculation
    x_start = np.array([0, 0, 1]).T
    x_end = np.array([img1_c, img1_r, 1]).T

    warped_x_start = np.matmul(warp_mat, x_start)
    warped_x_end = np.matmul(warp_mat, x_end)

    # W(x;p) coordinates
    warped_x_coordinate = np.linspace(warped_x_start[0], warped_x_end[0], img1_c)
    warped_y_coordinate = np.linspace(warped_x_start[1], warped_x_end[1], img1_r)
    warped_mesh_X, warped_mesh_Y = np.meshgrid(warped_x_coordinate, warped_y_coordinate)

    # img2 space coordinates
    img2_x_coordinate = np.linspace(0, img2_c - 1, img2_c)
    img2_y_coordinate = np.linspace(0, img2_r - 1, img2_r)

    img2_spline = RectBivariateSpline(img2_y_coordinate, img2_x_coordinate, img2)

    # Warp
    # I(W(x;p))
    IW = img2_spline.ev(warped_mesh_Y, warped_mesh_X)

    # Compute error image
    # img1(template) - IW_x
    residual = img1 - IW

    # matrix multiplication: (gradient)(jacobian)
    gradient_jacobian = np.zeros((img2_r * img2_c, 6))
    for y in range(img2_r):
        for x in range(img2_c):
            jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
            pixel = np.matmul(G[x + img2_c * y], jacobian)
            gradient_jacobian[x + img2_c * y] = pixel

    # back to original scale
    gradient_jacobian = gradient_jacobian * max_gradient

    # compute Hessian
    Hessian = np.matmul(gradient_jacobian.T, gradient_jacobian)

    # compute delta p
    dp = (
        np.matmul(
            np.matmul(np.linalg.inv(Hessian), gradient_jacobian.T),
            (residual).reshape((-1, 1)),
        )
    ).flatten()

    ### END CODE HERE ###
    return dp


def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    img1_r, img1_c = img1.shape
    img2_r, img2_c = img2.shape
    delta_p_norm = np.inf
    # find p
    delta_p = np.zeros(6)
    p = np.zeros(6)
    norm_threshold = 0.013
    while delta_p_norm > norm_threshold:
        delta_p = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += delta_p
        delta_p_norm = np.linalg.norm(delta_p)

    warp_mat = np.array(
        [[1 + p[0], 0 + p[2], 0 + p[4]], [0 + p[1], 1 + p[3], 0 + p[5]]]
    )

    # warp img1 -> range of warped coordinate
    x_start = np.array([0, 0, 1]).T
    x_end = np.array([img1_c, img1_r, 1]).T
    warped_x_start = np.matmul(warp_mat, x_start)
    warped_x_end = np.matmul(warp_mat, x_end)

    # W(x;p) coordinates
    warped_x_coordinate = np.linspace(warped_x_start[0], warped_x_end[0], img1_c)
    warped_y_coordinate = np.linspace(warped_x_start[1], warped_x_end[1], img1_r)
    warped_mesh_X, warped_mesh_Y = np.meshgrid(warped_x_coordinate, warped_y_coordinate)

    # Img2 space coordinates
    img2_x_coordinate = np.linspace(0, img2_c - 1, img2_c)
    img2_y_coordinate = np.linspace(0, img2_r - 1, img2_r)
    img2_spline = RectBivariateSpline(img2_y_coordinate, img2_x_coordinate, img2)

    # I(W(x;p))
    IW = img2_spline.ev(warped_mesh_Y, warped_mesh_X)

    # compute error image
    # template(img1) - IW
    residual = img1 - IW
    moving_image = np.abs(residual)
    th_hi = 0.25 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this

    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst


if __name__ == "__main__":
    data_dir = "data"
    video_path = "motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 150 / 20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
