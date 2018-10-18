import numpy as np

def contrast(matrix):
    height, width = matrix.shape
    sum_contrast = 0

    for i in range(height):
        for j in range(width):
            sum_contrast += ((abs(i - j))**2) * matrix[i, j]
    return sum_contrast

def energy(matrix):
    height, width = matrix.shape
    sum_energy = 0

    for i in range(height):
        for j in range(width):
            sum_energy += (matrix[i, j])**2
    return sum_energy

def homogeneity(matrix):
    height, width = matrix.shape
    sum_homogeneity = 0

    for i in range(height):
        for j in range(width):
            sum_homogeneity += matrix[i, j]/(1 + (abs(i - j)))
    return sum_homogeneity

def correlation(matrix):
    height, width = matrix.shape
    list_sum_i = []
    list_sum_j = []

    for i, j in zip(range(height), range(width)):
        sum_i = np.sum(matrix[i, :])
        sum_j = np.sum(matrix[:, j])
        list_sum_i.append(sum_i)
        list_sum_j.append(sum_j)
        sum_i = 0
        sum_j = 0

    average_sum_i = 0
    averagesum_j = 0

    for i in range(1, height+1):
        average_sum_i += i * (list_sum_i[i-1])
    for j in range(1, width+1):
        averagesum_j += j * (list_sum_j[j-1])

    sum_aux = 0
    for i in range(1, height+1):
        sum_aux += ((i - average_sum_i)**2) * list_sum_i[i - 1]
    standard_deviation_i = sum_aux**(1/2)
    sum_aux = 0
    for j in range(1, width+1):
        sum_aux += ((j - averagesum_j)**2) * list_sum_j[j - 1]
    standard_deviation_j = sum_aux**(1/2)

    sum_correlation = 0
    for i in range(1, height+1):
        for j in range(1, width+1):
            sum_correlation += ((i - average_sum_i) * (j - averagesum_j) * matrix[i-1, j-1])/\
                              (standard_deviation_i * standard_deviation_j)
    return sum_correlation

def all_features(matrix, exclude = 0):
    height, width = matrix.shape
    sum_contrast = 0
    sum_correlation = 0
    sum_energy = 0
    sum_homogeneity = 0
    list_sum_i = []
    list_sum_j = []

    for i, j in zip(range(height), range(width)):
        sum_i = np.sum(matrix[i, :])
        sum_j = np.sum(matrix[:, j])
        list_sum_i.append(sum_i)
        list_sum_j.append(sum_j)

    average_sum_i = 0
    average_sum_j = 0
    if exclude != 4:
        for i in range(1, height + 1):
            average_sum_i += i * (list_sum_i[i - 1])
        for j in range(1, width + 1):
            average_sum_j += j * (list_sum_j[j - 1])

        sum_aux = 0
        for i in range(1, height + 1):
            sum_aux += ((i - average_sum_i) ** 2) * list_sum_i[i - 1]
        standard_deviation_i = sum_aux ** (1 / 2)
        sum_aux = 0
        for j in range(1, width + 1):
            sum_aux += ((j - average_sum_j) ** 2) * list_sum_j[j - 1]
        standard_deviation_j = sum_aux ** (1 / 2)

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            if exclude != 1:
                sum_contrast += ((abs(i - j))**2) * matrix[i - 1, j - 1]
            if exclude != 2:
                sum_energy += (matrix[i - 1, j - 1])**2
            if exclude != 3:
                sum_homogeneity += matrix[i - 1, j - 1]/(1 + (abs(i - j)))
            if exclude != 4:
                sum_correlation += ((i - average_sum_i) * (j - average_sum_j) * matrix[i - 1, j - 1]) / \
                                      (standard_deviation_i * standard_deviation_j)

    return [sum_contrast, sum_correlation, sum_energy, sum_homogeneity]
