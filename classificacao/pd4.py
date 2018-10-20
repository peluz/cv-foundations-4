import cv2 as cv
import numpy as np
# import datetime

from feature_selection import feature_relation
from features_extraction import *
from glcm import *
from f_measure import f_measure
from bayes import *
from extract_interest import *
from sklearn import svm

def features_array(img_predio, img_nao_predio):
    matrix_glcm_p = glcm_mounter(img_predio)
    matrix_glcm_p /= np.sum(matrix_glcm_p)
    matrix_glcm_np = glcm_mounter(img_nao_predio)
    matrix_glcm_np /= np.sum(matrix_glcm_np)

    features_array = []
    features_array.append(all_features(matrix_glcm_p))
    features_array.append(all_features(matrix_glcm_np))
    features_array = np.array(features_array, dtype = np.float32)
    print(features_array)
    return features_array

#########################################################
# PROCESSAMENTO DAS GLCMS E SELEÇÃO DAS CARACTERÍSTICAS #
#########################################################
def main():
    img = cv.imread("data/vienna16.tif", -1)
    img_aux = cv.imread("data/gt/vienna16.tif", 0)
    # img_predio, img_nao_predio = extract(img, img_aux)
    # cv.imwrite("data/img_predio.tif", img_predio)
    # cv.imwrite("data/img_n_predio.tif", img_nao_predio)
    img_predio = cv.imread("data/img_predio_train.tif", 0)
    img_nao_predio = cv.imread("data/img_n_predio_train.tif", 0)
    train_features_array = features_array(img_predio, img_nao_predio)

    # Descubro qual feature devo ignorar #
    # feature_relation(train_features_array)

    img = cv.imread("data/vienna5.tif", -1)
    img_aux = cv.imread("data/gt/vienna5.tif", 0)
    img_predio = cv.imread("data/img_predio_teste.tif", 0)
    img_nao_predio = cv.imread("data/img_n_predio_teste.tif", 0)
    test_features_array = features_array(img_predio, img_nao_predio)
    
    img = cv.imread("data/vienna22.tif", -1)
    img_aux = cv.imread("data/gt/vienna22.tif", 0)
    img_predio = cv.imread("data/img_predio_final.tif", 0)
    img_nao_predio = cv.imread("data/img_n_predio_final.tif", 0)
    final_features_array = features_array(img_predio, img_nao_predio)
    
    #############################
    # CLASSIFICAÇÃO DAS IMAGENS #
    ##############################
    clf = svm.SVR(kernel='poly', C=1e3, degree=2)
    # clf = svm.SVC(kernel = 'poly', degree= 2)
    clf.fit(train_features_array, [0,1])
    print(clf.score(test_features_array, [0, 1]))
    print(clf.predict(test_features_array))
    clf.fit(test_features_array, [0,1])
    print(clf.predict(final_features_array))
    print(clf.score(test_features_array, [0, 1]))

if __name__ == '__main__':
    main()

