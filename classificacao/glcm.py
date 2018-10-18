import numpy as np

def glcm_hist_dictionary(image):
    height, width = image.shape
    list_histogram = []
    image_histogram = {}
    glcm_dictionary = {}
    for i, x in zip(range(height), range(height)):
        for j, y in zip(range(width), range(width - 1)):
            pixel_pair = (image[x, y], image[x, y + 1])
            if pixel_pair in glcm_dictionary:
                glcm_dictionary[pixel_pair] += 1
            else:
                glcm_dictionary[pixel_pair] = 1
            if image[i, j] in image_histogram:
                image_histogram[image[i, j]] += 1
            else:
                image_histogram[image[i, j]] = 1
                list_histogram.insert(0, image[i, j])
    list_histogram.sort()
    return [list_histogram, image_histogram, glcm_dictionary]

def glcm_mounter(image):
    list_histogram, image_histogram, glcm_dictionary = glcm_hist_dictionary(image)
    aux_shape = len(list_histogram)

    matrix_glcm = np.zeros((aux_shape, aux_shape), dtype=np.float32)

    for n, i in zip(list_histogram, range(aux_shape)):
        for m, j in zip(list_histogram, range(aux_shape)):
            if (n, m) in glcm_dictionary:
                # Talvez tenha que criar tuplas aqui se os índices da GLCM forem intensidades de pixels
                # matrix_glcm[i, j] = (n, m, glcm_dictionary[(n, m)]) -> tipo esse -> (i, j, glcm(i, j))
                matrix_glcm[i, j] = glcm_dictionary[(n, m)]

    return matrix_glcm

