import cv2 as cv
import numpy as np

from sklearn.cluster import KMeans


def main():
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init=10, max_iter=500, 
                    tol=0.0001, precompute_distances='auto', verbose=0,
                        random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')

    img = cv.imread("data/vienna16.tif", -1)
    img_gt = cv.imread("data/gt/vienna16.tif", 0)

    height = 1250
    width = 1250
    img = cv.resize(img, (height, width), interpolation = cv.INTER_CUBIC)    
    arr = []
    cnt_p = 0
    cnt_np = 0
    for h in range(height):
        for w in range(width):
            if(img_gt[h,w] == 255):
                arr.append(img[h, w, 0])
                arr.append(img[h, w, 1])
                arr.append(img[h, w, 2])
                cnt_p += 1
    for h in range(height):
        for w in range(width):
            if(img_gt[h,w] == 0):
                arr.append(img[h, w, 0])
                arr.append(img[h, w, 1])
                arr.append(img[h, w, 2])
                cnt_np += 1

    arr = np.array(arr, dtype = np.float32)
    arr = np.reshape(arr, (arr.shape[0]//3, 3))
    kmeans.fit(arr)

    ################
    # FINAL  IMAGE #
    ################
    img = cv.imread("data/vienna22.tif", -1)
    img_gt = cv.imread("data/gt/vienna22.tif", 0)
    img = cv.resize(img, (height,width), interpolation = cv.INTER_CUBIC)

    arr = []
    cnt_p = 0
    cnt_np = 0
    for h in range(height):
        for w in range(width):
            if(img_gt[h,w] == 255):
                arr.append(img[h, w, 0])
                arr.append(img[h, w, 1])
                arr.append(img[h, w, 2])
                
                cnt_p += 1

    for h in range(height):
        for w in range(width):
            if(img_gt[h,w] == 0):
                arr.append(img[h, w, 0])
                arr.append(img[h, w, 1])
                arr.append(img[h, w, 2])
                cnt_np += 1

    arr = np.array(arr, dtype = np.float32)
    arr = np.reshape(arr, (arr.shape[0]//3, 3))
    out = kmeans.predict(arr)

    hit = 0
    cnt = 0
    tp = 0
    fp = 0
    fn = 0
    for i in out:
        if(i == 1 and cnt < cnt_p):
            hit += 1
            tp += 1
        elif(i == 0 and cnt >= cnt_np):
            hit += 1
        elif(i == 1 and cnt >= cnt_np):
            fp += 1
        elif(i == 0 and cnt < cnt_p):
            fn += 1
        cnt += 1
    print("Accuracy: %f"%(hit/(cnt_p + cnt_np)))
    print("IoU: %f"%(tp/(tp+fn+fp)))
if __name__ == '__main__':
    main()