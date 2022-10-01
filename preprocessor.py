import math
import cv2
import numpy as np

from matplotlib import pyplot as plt

"""Funções destinadas para pre-processar imagens para detecção de pessoas"""


def get_variance(img):
    """Função destinada para encontrar a variância ao aplicar o algoritmo laplaciano"""
    return cv2.Laplacian(img, cv2.CV_64F).var()


def detect_borders(img):
    """Função destinada para aplicar o algoritmo de detecção de bordas implementado"""
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # equalized_img = cv2.equalizeHist(gray)
    # if get_variance(equalized_img) > 900:
    #     equalized_img = cv2.medianBlur(equalized_img, 23)
    # else:
    equalized_img = cv2.medianBlur(gray, 11)

    edges = cv2.Canny(equalized_img, 40, 100)

    cv2.imshow("OXE", edges)
    cv2.waitKey(0)

    edges = cv2.dilate(src=edges, kernel=np.ones((3, 3)), iterations=1)
    edges = cv2.erode(src=edges, kernel=np.ones((3, 3)), iterations=1)

    return edges


def improve_image(img):

    img = cv2.resize(img, (640, 480))

    original_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    min_hist, max_hist = min(hist)[0], max(hist)[0]
    contrast_decrease = (max_hist - min_hist)/2
    while contrast_decrease > 1:
        contrast_decrease = contrast_decrease/10

    img = (img * (1 - contrast_decrease)).astype(np.uint8)
    img = (img * 0.9).astype(np.uint8)

    # img = cv2.medianBlur(img, 17)

    kernel = np.array([
      [-1, -2, -1],
      [-2, 12, -2],
      [-1, -2, -1]
    ])
    borders = cv2.filter2D(img, -1, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    borders = cv2.dilate(borders, kernel, 0, None, 1)
    borders = cv2.erode(borders, kernel, 0, None, 1)


    low_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, low_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, low_kernel)

    bin_img = cv2.add(closed, borders)

    # cv2.imshow("THRESH MEAN",
    #            cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 97, 17))

    # _, otsu = cv2.threshold(bin_img, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, -1)
    # cv2.imshow("THRESH OTSU", otsu)
    # cv2.waitKey(0)
    #
    # _, binary = cv2.threshold(bin_img, 90, 255, cv2.THRESH_BINARY, -1)
    # cv2.imshow("THRESH", binary)
    # cv2.waitKey(0)
    #
    bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 97, 17)

    # bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 7)
    # _, bin_img = cv2.threshold(bin_img, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, -1)

    # bin_img = cv2.Canny(bin_img, 80, 255)
    bin_img = (255 - bin_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bin_img = cv2.dilate(bin_img, kernel, 0, None, 1)
    bin_img = cv2.erode(bin_img, kernel, 0, None, 1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # bin_img = cv2.dilate(bin_img, kernel, 0, None, 1)
    # bin_img = cv2.erode(bin_img, kernel, 0, None, 1)
    #

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(original_img, contours, -1, (255, 128, 255), -1)

    cropped_imgs = []
    for i in range(len(contours)):
        x, y, width, height = cv2.boundingRect(contours[i])
        if width > 250 and height > 250:
            cropped_imgs.append(original_img[y:y + height, x:x + width])


    # for i in range(1):
    #     x, y = contours[i].mean(axis=0)[0]
    #     x = math.ceil(x)
    #     y = math.ceil(y)
    #     cv2.circle(original_img, (x, y), 1, (0, 0, 255), 3)

    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    # sizes = stats[1:, -1]
    # nb_components = nb_components - 1
    #
    # for i in range(0, nb_components):
    #     if sizes[i] > 500:
    #         original_img[output == i + 1] = 255


    # cv2.imshow("Window", img2)
    # cv2.waitKey(0)

    return cropped_imgs


def find_objects(img):
    img = img.copy()

    ret, edges = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)
    edges = (255 - edges)

    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
    sizes = stats[:, -1]
    crops = []

    for i in range(0, nb_components):
        if sizes[i] >= 25000:
            crops.append([(stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])])

    return crops


def sliding_image(img, file_name, step_size=3):
    width = img.shape[1]
    height = img.shape[0]

    for i in range(step_size):

        aux = step_size - i

        if aux == 0:
            continue

        step_width = math.ceil(width / aux)
        step_height = math.ceil(height / aux)

        for x, y, image in sliding_window(img, (step_width, step_height)):

            clone = img.copy()
            objects = find_objects(image)

            cv2.rectangle(clone, (x, y), (x + step_width, y + step_height), (0, 255, 255), 2)
            cv2.imshow("window", clone)
            cv2.waitKey(1)

            if len(objects) > 0:
                new_file_path = "output\\{}_{}-{}-{}-{}.jpg".format(file_name[:-4], aux, len(objects), x, y)
                cv2.imwrite(new_file_path, image)


def sliding_window(image, window_size):
    for y in range(0, image.shape[0], window_size[1]):
        for x in range(0, image.shape[1], window_size[0]):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]