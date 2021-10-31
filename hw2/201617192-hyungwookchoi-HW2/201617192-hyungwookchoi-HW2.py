import math
import glob
import numpy as np
from PIL import Image, ImageDraw
import os
from matplotlib import pyplot as plt
import cv2

# parameters

datadir = "./data"
resultdir = "./results"
# you can calibrate these parameters (data dependent)
sigma = 9
highThreshold = 0.33
lowThreshold = 0.1
rhoRes = 1
thetaRes = math.pi / 180
nLines = 20


def replication_padding(Igs, size):
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
    padded_Igs = replication_padding(Igs, G.shape)

    # padded img size
    # pad_img_width, pad_img_height, pad_img_channels = reflected_img.shape
    # G size and padding size
    x_G, y_G = G.shape
    x_pad_size = x_G // 2
    y_pad_size = y_G // 2
    # original image matrix size (x, y, z)
    origin_rows, origin_cols = Igs.shape
    out_matrix = np.zeros((origin_rows, origin_cols))

    # 원래 이미지 기준에서의 픽셀 좌표
    origin_x, origin_y = 0, 0
    tmp_out = []

    while origin_x < origin_rows:
        re_x, re_y = origin_x + x_pad_size, origin_y + y_pad_size
        pix_result = []
        for r in range(-x_pad_size, x_pad_size + 1):
            for c in range(-y_pad_size, y_pad_size + 1):
                pix_result.append(
                    padded_Igs[re_x + r, re_y + c] * G[r + x_pad_size, c + y_pad_size]
                )
        tmp_out.append(sum(pix_result))
        origin_y += 1
        if origin_y >= origin_cols:
            origin_y = 0
            origin_x += 1
            if origin_x == origin_rows:
                out_matrix = np.array(tmp_out).reshape(origin_rows, origin_cols)

    return out_matrix  # Igs와 사이즈 동일


def gaussian_filter(input_image, size, sigmax, sigmay):
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
    x_filter_size, y_filter_size = size
    x_kernel = np.array(gaussian_kernel_1d(sigmax, x_filter_size)).reshape(
        x_filter_size, 1
    )
    y_kernel = np.array(gaussian_kernel_1d(sigmay, y_filter_size)).reshape(
        1, y_filter_size
    )

    gaussian_kernel_2D = np.dot(x_kernel, y_kernel)

    result = ConvFilter(input_image, gaussian_kernel_2D)
    return result


def sobel_filter(input_image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Sx = ConvFilter(input_image, sobel_x)
    Sy = ConvFilter(input_image, sobel_y)

    return Sx, Sy


def non_maximal_suppression(magnitude, angle):
    R, C = magnitude.shape

    suppressed_img = np.zeros((R, C), np.float64)

    for r in range(1, R - 1):
        for c in range(1, C - 1):
            # 각도를 0~180도로 만들어줌. 직선의 성질 이용.
            pix_angle = angle[r, c] if angle[r, c] > 0 else angle[r, c] + 1
            # 비교에 사용될 변수 선언
            i = 1.0
            j = 1.0
            # x 축 기준
            if (0 <= pix_angle < 1 / 8) or (7 / 8 <= pix_angle <= 1):
                i = magnitude[r, c - 1]
                j = magnitude[r, c + 1]

            elif 1 / 8 <= pix_angle < 3 / 8:
                i = magnitude[r + 1, c - 1]
                j = magnitude[r - 1, c + 1]
            elif 3 / 8 <= pix_angle < 5 / 8:
                i = magnitude[r - 1, c]
                j = magnitude[r + 1, c]
            elif 5 / 8 <= pix_angle < 7 / 8:
                i = magnitude[r - 1, c - 1]
                j = magnitude[r + 1, c + 1]

            if i < magnitude[r, c] and j < magnitude[r, c]:
                suppressed_img[r, c] = magnitude[r, c]
            else:
                suppressed_img[r, c] = 0
    return suppressed_img


def double_threshold(img, high_T, low_T):
    high = img.max() * high_T
    low = high * low_T

    def edge_tracking(img):
        R, C = img.shape
        for r in range(1, R):
            for c in range(1, C):
                if img[r, c] == (0.1):
                    around_pix = img[r - 1 : r + 2, c - 1 : c + 2]
                    if np.float64(1) in around_pix:
                        img[r, c] = 1.0
                    else:
                        img[r, c] = 0
        return img

    s_r, s_c = np.where(img >= high)
    w_r, w_c = np.where((low <= img) & (img < high))
    filtered_img = np.zeros(img.shape, np.float64)
    filtered_img[s_r, s_c] = 1.0
    filtered_img[w_r, w_c] = 0.1
    filtered_img = edge_tracking(filtered_img)
    return filtered_img


def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...
    g_size = (5, 5)
    gaussian_smoothed_img = gaussian_filter(
        Igs,
        g_size,
        sigma,
        sigma,
    )
    Ix, Iy = sobel_filter(gaussian_smoothed_img)
    Im = np.hypot(Ix, Iy)

    # Io range(-pi, pi)
    Io = np.arctan2(Iy, Ix)

    Io_div_pi = Io / np.pi

    suppressed_img = non_maximal_suppression(Im, Io_div_pi)
    Im = double_threshold(suppressed_img, highThreshold, lowThreshold)
    return Im, Io, Ix, Iy


"""
H is the Hough transform accumulator that contains the number of ‘votes’ for all the possible lines passing through the image.
Im: edge magnitude
rhoRes: 허프 스페이스 y축 확대 비율
thetaRes: x축 확대 비율
"""


def HoughTransform(Im, rhoRes, thetaRes):
    # 세타 축에 해당
    space_theta = np.deg2rad(np.linspace(-90, 90, num=int(math.pi // thetaRes) + 1))

    R, C = Im.shape

    # x,y 좌표에 따라 rho가 가질 수 있는 가장 큰 값. 사진의 대각선에 해당.
    r_max = int(math.sqrt(R ** 2 + C ** 2))

    # 좌표계 선언 (y축, 즉 rho의 최댓값을 확대율로 나눈의 2배(음수인 값을 모두 양수로, x축: 2pi를 세타의 확대율로 나눈 값 ))
    # 여기서 x축 y축은 출력될 그림을 기준으로 함.
    H = np.zeros((2 * r_max // rhoRes, len(space_theta)), dtype=np.uint64)

    for x in range(C):
        for y in range(R):
            if Im[y, x] != 0:
                for i, theta in enumerate(space_theta):
                    # rho 수식에 맞게 계산 후, 양의 방향으로 최댓값만큼 평행 이동(사진에 표현하기 위해), 그 후 확대율로 나눠서 좌표계에 맞는 rho좌표 구하기
                    r = round(x * np.cos(theta) + y * np.sin(theta)) + r_max
                    # 이미지 상에서 x, y축을 맞추기 위해 H[rho(y축), theta(x축)]에 VOTE를 count 함.
                    H[r // rhoRes, i] += 1

    return H


def HoughLines(H, rhoRes, thetaRes, nLines, img_size):
    # reference: https://gist.github.com/ri-sh/45cb32dd5c1485e273ab81468e531f09

    copy_of_H = np.copy(H)
    img_R, img_C = img_size
    rho_arr = []
    theta_arr = []
    R, C = H.shape
    nh_size_x, nh_size_y = int(30), int(30)

    for _ in range(nLines):
        index_of_max = np.argmax(copy_of_H)
        y_idx, x_idx = np.unravel_index(index_of_max, (R, C))
        theta_arr.append(x_idx)
        rho_arr.append(y_idx)
        # if x_idx is too close to the edges choose appropriate values
        if (x_idx - nh_size_x / 2) < 0:
            min_x = 0
        else:
            min_x = x_idx - (nh_size_x / 2)
        if x_idx + nh_size_x / 2 + 1 > C:
            max_x = C
        else:
            max_x = x_idx + nh_size_x / 2 + 1

        if (y_idx - nh_size_y / 2) < 0:
            min_y = 0
        else:
            min_y = y_idx - (nh_size_y / 2)
        if y_idx + nh_size_y / 2 + 1 > R:
            max_y = R
        else:
            max_y = y_idx + nh_size_y / 2 + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(np.ceil(min_x)), int(max_x)):
            for y in range(int(np.ceil(min_y)), int(max_y)):
                # remove neighborhoods in copy_of_H
                copy_of_H[y, x] = 0

    lTheta = np.array(theta_arr)

    # restore integer level to radian(  90 deg shifted) and rad to deg
    # 음이 아닌 정수로 표현되는 좌표값을 라디안으로 변환 후, 90도 평행 이동 했던 것을 빼준다.
    lTheta = lTheta * thetaRes / (math.pi / 180) - 90
    diag = int(np.ceil(math.sqrt(img_R ** 2 + img_C ** 2)))
    lRho = np.array([rho * rhoRes - diag for rho in rho_arr])
    return lRho, lTheta


def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...
    l = 0
    return l


def main():
    global filename

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    # read images
    for img_path in glob.glob(datadir + "/*.jpg"):
        filename = img_path.split("/")[-1][0:-4]

        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.0

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(
            Igs,
            sigma,
            highThreshold,
            lowThreshold,
        )
        R, C = Im.shape

        H = HoughTransform(Im, rhoRes, thetaRes)
        img_H = Image.fromarray(np.uint8(H), "L")
        img_H.save(os.path.join(resultdir, f"{filename}_H.jpg"))

        lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines, (R, C))
        image_to_save = Image.fromarray(np.uint8(Im * 255), "L")
        image_to_save.save(os.path.join(resultdir, f"{filename}_Im.jpg"))

        # 원본 이미지
        rgb_img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(rgb_img)

        # 원본 이미지에 허프라인 그리기
        for n in range(nLines):
            deg_theta = np.deg2rad(lTheta)
            cos_t = np.cos(deg_theta)
            sin_t = np.sin(deg_theta)

            origin_rho = lRho[n]
            # hor
            if lTheta[n] == 0:
                y = round((origin_rho))
                draw.line((0, y, C - 1, y), fill="red")
            else:
                # 사진 왼쪽 y축과 만나는 경우
                x1 = 0
                y1 = round((origin_rho) / sin_t[n])
                # 사진 우측의 축과 만나는 경우
                x2 = C - 1
                y2 = round((origin_rho - x2 * cos_t[n]) / sin_t[n])
                draw.line((x1, y1, x2, y2), fill="red")

        rgb_img.show()
        rgb_img.save(os.path.join(resultdir, f"{filename}_HoughLines.jpg"))

        l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == "__main__":
    main()
