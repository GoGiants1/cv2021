import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def compute_h(p1, p2):
    """
    Inputs
        p1: (N,2) matrix
            coordinates in Image
        p2: (N,2) matrix
            coordinates in Image
    Outputs
        H: (3,3) matrix
            linear eqn (from p2 to p1)
    Hint
        H will only be determined up to scale, and the solution will involve an SVD

        SVD function: Can Use np.linalg.svd, scipy.linalg.svd etc.

    """
    H = np.zeros((3, 3))
    N, _ = p1.shape

    # A matrix. ref: theory Question 3 (a)
    A = np.zeros((1, 9))
    for i in range(N):
        xi = p1[i, 0]
        yi = p1[i, 1]
        Xi = p2[i, 0]
        Yi = p2[i, 1]
        first_row = [Xi, Yi, 1, 0, 0, 0, -xi * Xi, -xi * Yi, -xi]
        second_row = [0, 0, 0, Xi, Yi, 1, -yi * Xi, -yi * Yi, -yi]
        two_rows = np.vstack((first_row, second_row))
        if i == 0:
            A = two_rows
        else:
            A = np.vstack((A, two_rows))

    _, Sig, V_transposed = np.linalg.svd(A)
    V = V_transposed.T

    # 이론문제에서 풀었던 문제처럼 Sigma 행렬의 대각 성분 중 가장 작은 eigenvalue를 찾아서, 해당하는 인덱스의 V의 column을 찾는다. 이 열 벡터가 우리가 찾는 h임.
    # 이후 h 벡터를 H(행렬)로 변환.
    v_col = V[:, -1]
    H = np.reshape(v_col, (3, 3))
    return H


def compute_h_norm(p1, p2):
    """
    Inputs
        p1: (N,2) matrix
            coordinates in Image
        p2: (N,2) matrix
            coordinates in Image
    Outputs
        H: (3,3) matrix
            normalized H matrix
    Hint
        Express the normalization as a matrix.

    Normalize p1, p2 and compute_h
    So that, improve numerical stability.
    정규화: 값에 평균값 뺴고, 표준 편차를 나누기?
    p1 ≡ Hp2
    """
    # normalization matrix
    N, _ = p1.shape

    max_p1, min_p1 = np.max(p1), np.min(p1)
    normalization_mat_p1 = np.array(
        [[1.0, 0.0, -min_p1], [0.0, 1.0, -min_p1], [0.0, 0.0, (max_p1 - min_p1)]]
    ) / (max_p1 - min_p1)
    max_p2, min_p2 = np.max(p2), np.min(p2)
    normalization_mat_p2 = np.array(
        [[1.0, 0.0, -min_p2], [0.0, 1.0, -min_p2], [0.0, 0.0, (max_p2 - min_p2)]]
    ) / (max_p2 - min_p2)
    normalized_p1 = np.zeros(p1.shape)
    normalized_p2 = np.zeros(p2.shape)
    for i in range(N):
        x_1, y_1 = p1[i]
        x_2, y_2 = p2[i]
        homogeneous_p1 = np.array([x_1, y_1, 1.0])
        homogeneous_p2 = np.array([x_2, y_2, 1.0])
        normalized_p1[i] = (normalization_mat_p1 @ homogeneous_p1.T)[:2]
        normalized_p2[i] = (normalization_mat_p2 @ homogeneous_p2.T)[:2]

    H_normalized = compute_h(normalized_p1, normalized_p2)

    # 정규화 해서 구했던 H를 원래 p1를 구할 수 있도록 변형
    normalization_mat_p1_inverse = np.linalg.inv(normalization_mat_p1)
    norm_mat_inv_mul_H = normalization_mat_p1_inverse @ H_normalized
    H = norm_mat_inv_mul_H @ normalization_mat_p2

    return H


def warp_image(igs_in, igs_ref, H):
    """
    inverse  warping
    H^-1
    """
    R_in, C_in, _ = igs_in.shape
    R_ref, C_ref, _ = igs_ref.shape
    H_inverse = np.linalg.inv(H)

    # full size warpped image
    # igs_warp = np.zeros((2400,1600,3), dtype=np.uint8)

    # proto 2 size
    igs_warp = np.zeros((1200, 1600, 3), dtype=np.uint8)
    igs_merge = np.pad(
        igs_ref,
        ((600, 600), (1600, 0), (0, 0)),
        mode="constant",
    )

    for r in range(-600, 1800):
        for c in range(-1600, 1600):
            # inverse warping (in -> ref)
            igs_merge_coordinate = np.array([c, r, 1])
            igs_in_coordinate_homo = H_inverse @ igs_merge_coordinate.T
            in_y, in_x, in_z = igs_in_coordinate_homo
            in_x, in_y = in_x / in_z, in_y / in_z

            # bilinear 보간(interpolation)
            if 0 <= in_x < R_in and 0 <= in_y < C_in:
                org_r, org_c = int(np.floor(in_x)), int(np.floor(in_y))

                # (r,c) 기준으로 어디에 위치해 있는지 0~1 스케일로 표현 => (a,b)
                a = in_x - np.floor(in_x)
                b = in_y - np.floor(in_y)
                if org_r + 1 < R_in and org_c + 1 < C_in:
                    rc = igs_in[org_r, org_c]
                    r1_c = igs_in[org_r + 1, org_c]
                    r_c1 = igs_in[org_r, org_c + 1]
                    r1_c1 = igs_in[org_r + 1, org_c + 1]

                    interpolated_pixel = (
                        (1 - a) * (1 - b) * rc
                        + a * (1 - b) * r1_c
                        + a * b * r1_c1
                        + (1 - a) * b * r_c1
                    )

                    igs_merge[r + 600, c + 1600] = interpolated_pixel
                    # full size warpped image
                    # igs_warp[r+600,c +1600] = interpolated_pixel

                    # Case warpped size = proto 2 size
                    if (0 <= (r) < R_ref) and (0 <= c < C_ref):
                        igs_warp[r, c] = interpolated_pixel

    return igs_warp, igs_merge


def rectify(igs, p1, p2):
    # TODO ...
    H = compute_h_norm(p2, p1)
    H_inverse = np.linalg.inv(H)

    igs_rec = np.zeros(igs.shape, np.uint8)

    R, C, _ = igs.shape

    for r in range(R):
        for c in range(C):
            # inverse warping (in -> ref)
            igs_ref_coordinate = np.array([c, r, 1])
            igs_in_coordinate_homo = H_inverse @ igs_ref_coordinate.T
            in_y, in_x, in_z = igs_in_coordinate_homo
            in_x, in_y = in_x / in_z, in_y / in_z

            # bilinear 보간(interpolation)
            if 0 <= in_x < R and 0 <= in_y < C:
                org_r, org_c = int(np.floor(in_x)), int(np.floor(in_y))

                # (r,c) 기준으로 어디에 위치해 있는지 0~1 스케일로 표현 => (a,b)
                a = in_x - np.floor(in_x)
                b = in_y - np.floor(in_y)
                if org_r + 1 < R and org_c + 1 < C:
                    rc = igs[org_r, org_c]
                    r1_c = igs[org_r + 1, org_c]
                    r_c1 = igs[org_r, org_c + 1]
                    r1_c1 = igs[org_r + 1, org_c + 1]

                    interpolated_pixel = (
                        (1 - a) * (1 - b) * rc
                        + a * (1 - b) * r1_c
                        + a * b * r1_c1
                        + (1 - a) * b * r_c1
                    )
                    igs_rec[r, c] = interpolated_pixel

    return igs_rec


def set_cor_mosaic():
    """
    H 사이즈 3 x 3
    min correspond pair: 4
    (x,y)
    """
    # TODO 1
    p_in = np.array(
        [[1119, 765], [1254, 964], [980, 451], [1284, 514], [1282, 416], [1258, 257]]
    )
    p_ref = np.array(
        [[373, 769], [509, 954], [224, 440], [536, 517], [535, 423], [512, 268]]
    )

    return p_in, p_ref


def set_cor_rec():
    """
    (x,y): 이미지 상에서 가로, 세로
    """
    c_in = np.array(
        [[1063, 166], [1222, 150], [1384, 133], [1051, 852], [1214, 859], [1389, 866]]
    )
    c_ref = np.array(
        [[1063, 166], [1222, 166], [1384, 166], [1051, 852], [1214, 852], [1389, 866]]
    )

    return c_in, c_ref


def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open("data/porto1.png").convert("RGB")
    img_ref = Image.open("data/porto2.png").convert("RGB")

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save("porto1_warped.png")
    img_merge.save("porto_merged.png")
    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open("data/iphone.png").convert("RGB")
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save("iphone_rectified.png")


if __name__ == "__main__":
    main()
