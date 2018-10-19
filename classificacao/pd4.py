import cv2 as cv
import numpy as np
# import datetime

from feature_selection import feature_relation
from features_extraction import *
from glcm import *
from f_measure import f_measure


#########################################################
# PROCESSAMENTO DAS GLCMS E SELEÇÃO DAS CARACTERÍSTICAS #
#########################################################

img = cv.imread("data/vienna16.tif", -1)
height, width, channels = img.shape
print(height, width)
train_features_array = []
print("\nProcessing...\n")
# Para cada imagem eu a transformo em cinza, adquiro sua matriz GLCM e suas respectivas features,
# então as guardo em um Array 
image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

matrix_glcm = glcm_mounter(image)
matrix_glcm /= np.sum(matrix_glcm)
print(matrix_glcm)
print(np.shape(matrix_glcm))

features = all_features(matrix_glcm)
train_features_array.append(features)

train_features_array = np.array(train_features_array, dtype = np.float32)
print(train_features_array)
print(np.shape(train_features_array))

# Descubro qual feature devo ignorar #
feature_relation(train_features_array)

image = cv.imread("data/vienna5.tif", -1)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
test_features_array = []
matrix_glcm = glcm_mounter(image)
matrix_glcm /= np.sum(matrix_glcm)

features = all_features(matrix_glcm, 3) # Não calculo a feature eliminada #
test_features_array.append(features)

test_features_array = np.array(test_features_array, dtype = np.float32)
print(test_features_array)
print(np.shape(test_features_array))

######################################################
# CÁLCULO DAS DISTÂNCIAS E CLASSIFICAÇÃO DAS IMAGENS #
######################################################


