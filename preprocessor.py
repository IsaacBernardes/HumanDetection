import cv2
import numpy as np

"""Funções destinadas para pre-processar imagens para detecção de pessoas"""


def get_variance(img):
    """Função destinada para encontrar a variância ao aplicar o algoritmo laplaciano"""
    return cv2.Laplacian(img, cv2.CV_64F).var()


def detect_borders(input_path, output_path):
    """Função destinada para aplicar o algoritmo de detecção de bordas implementado"""
    img = cv2.imread(input_path, 0)
    equalized_img = cv2.equalizeHist(img)

    if get_variance(img) > 30:
        equalized_img = cv2.GaussianBlur(equalized_img, (5, 5), sigmaX=0, sigmaY=0)
    else:
        equalized_img = cv2.GaussianBlur(equalized_img, (13, 13), sigmaX=0, sigmaY=0)

    filter_mask = np.array([[-2, -1, 0], [-1, 8, -1], [0, -1, -2]])
    img_borders = cv2.filter2D(equalized_img, kernel=filter_mask, ddepth=0)

    cv2.imwrite(output_path, img_borders)


