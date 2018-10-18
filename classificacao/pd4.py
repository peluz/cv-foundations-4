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
train_features_array = []
print("\nProcessing...\n")
# Para cada imagem eu a transformo em cinza, adquiro sua matriz GLCM e suas respectivas features, então as guardo em um Array 
image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

matrix_glcm = glcm_mounter(image)
matrix_glcm /= np.sum(matrix_glcm)

features = all_features(matrix_glcm)
train_features_array.append(features)

train_features_array = np.array(train_features_array, dtype = np.float32)
print(train_features_array)

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

######################################################
# CÁLCULO DAS DISTÂNCIAS E CLASSIFICAÇÃO DAS IMAGENS #
######################################################

# confusion_matrix = np.zeros((2, 2), dtype=np.uint8)

# height, width = test_features_array.shape

# distance_list = []
# aux_list = []

# for i in range(height):
#     # Supondo que distancia = (Δx² + Δy² + Δz²)^(1/2)
#     # Crio um array do tipo [Δx1²   Δy1²   Δz1² ]
#     #                               :
#     #                               :
#     #                       [Δx75²  Δy75²  Δz75²]
#     result_features_array = (train_features_array - test_features_array[i, :]) ** 2
#     # Crio uma flag pra saber o tipo real da imagem
#     if i < 25:
#         flag_true = 0
#     elif i > 25 and i < 51:
#         flag_true = 1
#     else:
#         flag_true = 2

#     for j in range(height):
#     # Crio outra flag pra saber o tipo que a imagem foi classificada
#         if j < 25:
#             flag = 0
#         elif j > 25 and j < 51:
#             flag = 1
#         else:
#             flag = 2
#         distance = (np.sum(result_features_array[j, :3])) ** (1 / 2) # Calculo a raiz quadrada da soma dos meus Δ
#         distance_list.append((distance, flag)) # Marco com uma flag pra saber de qual conjunto é essa distância #
#     distanceAsphalt = distance_list[:25]
#     distanceDanger = distance_list[25:50]
#     distanceGrass = distance_list[50:]

#     distanceAsphalt.sort()
#     distanceDanger.sort()
#     distanceGrass.sort()
#     distance_list.sort()

#     asphaltCounter = 0
#     dangerCounter = 0
#     grassCounter = 0
#     # Com os ifs consigo selecionar o melhor cenário de vizinhos para cada tipo de classe
#     if flag_true == 0:
#         for counter in range(15): # Vou contando o número de vizinhos perto do ponto que está sendo classificado #
#             if distance_list[counter][1] == 0:
#                 asphaltCounter += 1
#             elif distance_list[counter][1] == 1:
#                 dangerCounter += 1
#             elif distance_list[counter][1] == 2:
#                 grassCounter += 1
#         aux_list = [(asphaltCounter, 0), (dangerCounter, 1), (grassCounter, 2)]
#         aux_list.sort()
#         confusion_matrix[aux_list[2][1], flag_true] += 1 # Marco isso na matriz com a flag que eu classifiquei e a flag de verdade #
#         distance_list = []
#     elif flag_true == 1:
#         for counter in range(8):  # Vou contando o número de vizinhos perto do ponto que está sendo classificado #
#             if distance_list[counter][1] == 0:
#                 asphaltCounter += 1
#             elif distance_list[counter][1] == 1:
#                 dangerCounter += 1
#             elif distance_list[counter][1] == 2:
#                 grassCounter += 1
#         aux_list = [(asphaltCounter, 0), (dangerCounter, 1), (grassCounter, 2)]
#         aux_list.sort()
#         confusion_matrix[aux_list[2][1], flag_true] += 1  # Marco isso na matriz com a flag que eu classifiquei e a flag de verdade #
#         distance_list = []
#     else:
#         for counter in range(9):  # Vou contando o número de vizinhos perto do ponto que está sendo classificado #
#             if distance_list[counter][1] == 0:
#                 asphaltCounter += 1
#             elif distance_list[counter][1] == 1:
#                 dangerCounter += 1
#             elif distance_list[counter][1] == 2:
#                 grassCounter += 1
#         aux_list = [(asphaltCounter, 0), (dangerCounter, 1), (grassCounter, 2)]
#         aux_list.sort()
#         confusion_matrix[aux_list[2][1], flag_true] += 1  # Marco isso na matriz com a flag que eu classifiquei e a flag de verdade #
#         distance_list = []
# safety_matrix, f_measure_safe, f_measure_unsafe = f_measure(confusion_matrix)

