import cv2 as cv
import numpy as np

def largura_altura(pixels_predios):
    alt = int(pixels_predios ** 0.5)

    while (pixels_predios % alt != 0) :
        alt -= 1
    
    return alt, pixels_predios // alt

def extract(img, img_aux):
    height, width, channels = img.shape
    print(height, width)
    print("\nProcessing...\n")
    # Para cada imagem eu a transformo em cinza, adquiro sua matriz GLCM e suas respectivas features,
    # então as guardo em um Array 
    img_aux = cv.imread("data/gt/vienna16.tif", 0)
    pixels_predios = 0
    for y in range(height):
        for i in range(width):
            if(img_aux[y, i] == 255):
                pixels_predios += 1

    alt, larg = largura_altura(pixels_predios)
    img_predio = np.zeros((larg, alt, channels), dtype = np.uint8) - 1
    img_nao_predio = np.zeros((width - larg, height - alt, channels), dtype = np.uint8) - 1
    for y in range(height):
        for i in range(width):
            if(img_aux[y, i] == 255):
                if(img_predio[y % larg, i % alt, 0] != -1):
                    img_predio[y % larg, i % alt, :] = img[y, i, :]
                    # img_predio[y % larg, i % alt, 0] = img[y, i, 0]
                    # img_predio[y % larg, i % alt, 1] = img[y, i, 1]
                    # img_predio[y % larg, i % alt, 2] = img[y, i, 2]
                else:
                    for h in range(alt):
                        for w in range(larg):
                            if(img_predio[h, w, 0] != -1):
                                img_predio[h, w, :] = img[y, i, :]
                                # img_predio[h, w, 0] = img[y, i, 0]
                                # img_predio[h, w, 1] = img[y, i, 1]
                                # img_predio[h, w, 2] = img[y, i, 2]
                                w = larg
                                h = alt
            else:
                if(img_nao_predio[y % (width - larg), i % (height - alt), 0] != -1):
                    img_nao_predio[y % (width - larg), i % (height - alt), :] = img[y, i, :]
                    # img_nao_predio[y % larg, i % alt, 0] = img[y, i, 0]
                    # img_nao_predio[y % larg, i % alt, 1] = img[y, i, 1]
                    # img_nao_predio[y % larg, i % alt, 2] = img[y, i, 2]
                else:
                    for h in range(height - alt):
                        for w in range(width - larg):
                            if(img_nao_predio[h, w, 0] != -1):
                                img_nao_predio[h, w, :] = img[y, i, :]
                                # img_nao_predio[h, w, 0] = img[y, i, 0]
                                # img_nao_predio[h, w, 1] = img[y, i, 1]
                                # img_nao_predio[h, w, 2] = img[y, i, 2]
                                w = height
                                h = width
    return img_predio, img_nao_predio