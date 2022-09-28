import math
import cv2
import numpy as np

"""Funções destinadas para pre-processar imagens para detecção de pessoas"""


def get_variance(img):
    """Função destinada para encontrar a variância ao aplicar o algoritmo laplaciano"""
    return cv2.Laplacian(img, cv2.CV_64F).var()


def detect_borders(img):
    """Função destinada para aplicar o algoritmo de detecção de bordas implementado"""
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    equalized_img = cv2.equalizeHist(gray)
    if get_variance(equalized_img) > 900:
        equalized_img = cv2.medianBlur(equalized_img, 23)
    else:
        equalized_img = cv2.medianBlur(equalized_img, 31)

    edges = cv2.Canny(equalized_img, 40, 100)
    edges = cv2.dilate(src=edges, kernel=np.ones((3, 3)), iterations=1)
    edges = cv2.erode(src=edges, kernel=np.ones((3, 3)), iterations=1)

    return edges


def find_objects(img):
    img = img.copy()
    img = cv2.medianBlur(img, 17)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)
    edges = (255 - edges)

    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
    sizes = stats[:, -1]
    crops = []

    for i in range(0, nb_components):
        if sizes[i] >= 25000:
            crops.append([(stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])])

    return crops


def sliding_image(input_path, file_name, step_size=3):
    img = cv2.imread(input_path)

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