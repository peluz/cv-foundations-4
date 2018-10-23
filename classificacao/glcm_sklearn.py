import cv2 as cv
import itertools
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import svm

# 32203917/3 p
# 42796083/3 np


def largura_altura(pixels_predios):
    alt = int(pixels_predios ** 0.5)

    while (pixels_predios % alt != 0) :
        alt -= 1
    
    return alt, pixels_predios // alt

def main():
    img = cv.imread("data/vienna16.tif", -1)
    img_gt = cv.imread("data/gt/vienna16.tif", -1)
    height, width, channels = img.shape
    arr_p = []
    arr_np = []
    cnt_p = 0
    cnt_np = 0
    for y in range(height):
        for x in range(width):
            if(img_gt[y,x] == 255):
                arr_p.append(img[y, x, 0])
                arr_p.append(img[y, x, 1])
                arr_p.append(img[y, x, 2])
                cnt_p += 3
            else:
                arr_np.append(img[y, x, 0])
                arr_np.append(img[y, x, 1])
                arr_np.append(img[y, x, 2])
                cnt_np += 3


    alt_p, larg_p = largura_altura(cnt_p)
    arr_p = np.array(arr_p, dtype = np.uint8)
    arr_p = np.reshape(arr_p, (alt_p, larg_p))  

    alt_np, larg_np = largura_altura(cnt_np)
    arr_np = np.array(arr_np, dtype = np.uint8)
    arr_np = np.reshape(arr_np, (alt_np, larg_np))
    print(arr_p[:20])
    print(arr_np[:20])

    # cv.imwrite("predios.tif", arr_p)
    # cv.imwrite("nao_predios.tif", arr_np)
    
    features = ['contrast', 'homogeneity']
    build = []
    n_build = []
    build_feat = []
    n_build_feat = []
    glcm_p = greycomatrix(arr_p, [1,2,3,4], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    # glcm_p = greycomatrix(arr_p, [1], [0], 256, symmetric=True, normed=True)
    # glcm_np = greycomatrix(arr_p, [1], [0], 256, symmetric=True, normed=True)
    glcm_np = greycomatrix(arr_np, [1,2,3,4], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    for feat in features:
        build.append(np.mean(greycoprops(glcm_p, feat)))
        n_build.append(np.mean(greycoprops(glcm_np, feat)))
        build_feat.append(build[:])
        n_build_feat.append(n_build[:])
        build.clear()
        n_build.clear()

    build_feat = np.transpose(np.asarray(build_feat))
    n_build_feat = np.transpose(np.asarray(n_build_feat))

    np.random.shuffle(build_feat)
    np.random.shuffle(n_build_feat)

    build_feat = normalize(build_feat)
    n_build_feat = normalize(n_build_feat)
    print(build_feat.shape)
    print(n_build_feat.shape)
    
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=1, p=2, weights='uniform')
    X_train = np.append(build_feat, n_build_feat)
    X_train = np.reshape(X_train, (X_train.shape[0]//len(features), len(features)))
    print(X_train)
    print(X_train.shape)
    knn.fit( X_train, np.repeat(np.arange(2),1) )

    ################
    # IMAGEM FINAL #
    ################
    img = cv.imread("data/vienna22.tif", 0)
    img_gt = cv.imread("data/gt/vienna22.tif", -1)

    feat = []
    features = ['contrast', 'homogeneity']
    cnt = 0
    print("entrou")
    for y in range(0, height, 25):
        for x in range(0, width, 25):
            arr = img[y:y+25, x:x+25]
            glcm = greycomatrix(arr, [1,2,3,4], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
            for f in features:
                feat.append(np.mean(greycoprops(glcm, f)))
            cnt += 1
    
    print("saiu do for")
    feat = np.array(feat)
    feat = feat.reshape(cnt, 2)
    feat = normalize(feat)
    label = knn.predict(feat)
    img_show = np.copy(img_gt)
    hit = 0
    cnt = 0
    tp = 0
    fp = 0
    fn = 0
    cnt = 0
    print("entrou no ultimo")
    for y in range(0, height, 25):
        for x in range(0, width, 25):
            if(label[cnt] == 1):
                img_show[y:y+25, x:x+25] = 0
            else:
                img_show[y:y+25, x:x+25] = 255
            for h in range(y, y+25):
                for w in range(x, x+25):
                    if(img_gt[h, w] == 255):
                        if(label[cnt] == 0): # É prédio e o knn previu como prédio
                            hit += 1
                            tp += 1
                        else: # Incorretamente previu que não é um prédio
                            fn += 1
                    else:
                        if(label[cnt] == 0): # Incorretamente previu que é um prédio
                            fp += 1
                        else: # Não é prédio e previu que não é prédio mesmo
                            hit += 1
            cnt += 1

    print("Accuracy: %f"%(hit/(height*width)))
    print("IoU: %f"%(tp/(tp+fn+fp)))
    img_show = img_show.astype(np.uint8)
    cv.imwrite("prev_teste.tif", img_show)

if __name__ == '__main__':
    main()