import numpy as np

def f_measure(confusion_matrix):
    safety_matrix = np.zeros((2, 2), dtype=np.uint8)

    safety_matrix[0, 0] = confusion_matrix[0, 0]
    safety_matrix[0, 1] = confusion_matrix[0, 1] + confusion_matrix[0, 2]
    safety_matrix[1, 0] = confusion_matrix[1, 0] + confusion_matrix[2, 0]
    safety_matrix[1, 1] = confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]

    precision_safe = safety_matrix[0, 0]
    precision_safe /= float(np.sum(safety_matrix[0]))

    recall_safe = safety_matrix[0, 0]
    recall_safe /= float(np.sum(safety_matrix[:, 0]))

    precision_unsafe = safety_matrix[1, 1]
    precision_unsafe /= float(np.sum(safety_matrix[1]))

    recall_unsafe = safety_matrix[1, 1]
    recall_unsafe /= float(np.sum(safety_matrix[:, 1]))

    f_measure_safe = 2 * (precision_safe * recall_safe) / (precision_safe + recall_safe)
    f_measure_unsafe = 2 * (precision_unsafe * recall_unsafe) / (precision_unsafe + recall_unsafe)

    return [safety_matrix, f_measure_safe, f_measure_unsafe]
