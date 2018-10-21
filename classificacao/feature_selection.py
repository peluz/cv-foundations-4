import numpy as np
import matplotlib.pyplot as plt

# Faço a relação de todas as features entre os três tipos de terreno #
# 1 = Asfalto, primeiras 25 imagens #
# 2 = Perigo, próximas 25 imagens #
# 3 = Grama, próximas 25 imagens #
# Com esses valores, seleciono o maior valor de cada terreno e identifico a característica em comum nas 3 relações, então a elimino #

def feature_relation(features_array):
    for type in range(3):
        if type == 1:
            relation_con_cor = np.corrcoef(features_array[:25, 0], features_array[:25, 1])
            relation_con_ene = np.corrcoef(features_array[:25, 0], features_array[:25, 2])
            relation_con_hom = np.corrcoef(features_array[:25, 0], features_array[:25, 3])
            relation_cor_ene = np.corrcoef(features_array[:25, 1], features_array[:25, 2])
            relation_cor_hom = np.corrcoef(features_array[:25, 1], features_array[:25, 3])
            relationenehom = np.corrcoef(features_array[:25, 2], features_array[:25, 3])
            relation_list_1 = [(abs(relation_con_cor[0, 1]), 'Contrast&Correlation'), (abs(relation_con_ene[0, 1]), 'Contrast&energy'), (abs(relation_con_hom[0, 1]), 'Contrast&homogeneity'),
                             (abs(relation_cor_ene[0, 1]), 'Correlation&energy'), (abs(relation_cor_hom[0, 1]), 'Correlation&homogeneity'), (abs(relationenehom[0, 1]), 'energy&homogeneity')]
            relation_list_1.sort()
        elif type == 2:
            relation_con_cor = np.corrcoef(features_array[25:50, 0], features_array[25:50, 1])
            relation_con_ene = np.corrcoef(features_array[25:50, 0], features_array[25:50, 2])
            relation_con_hom = np.corrcoef(features_array[25:50, 0], features_array[25:50, 3])
            relation_cor_ene = np.corrcoef(features_array[25:50, 1], features_array[25:50, 2])
            relation_cor_hom = np.corrcoef(features_array[25:50, 1], features_array[25:50, 3])
            relationenehom = np.corrcoef(features_array[25:50, 2], features_array[25:50, 3])
            relation_list_2 = [(abs(relation_con_cor[0, 1]), 'Contrast&Correlation'), (abs(relation_con_ene[0, 1]), 'Contrast&energy'), (abs(relation_con_hom[0, 1]), 'Contrast&homogeneity'),
                             (abs(relation_cor_ene[0, 1]), 'Correlation&energy'), (abs(relation_cor_hom[0, 1]), 'Correlation&homogeneity'), (abs(relationenehom[0, 1]), 'energy&homogeneity')]
            relation_list_2.sort()
        else:
            relation_con_cor = np.corrcoef(features_array[50:, 0], features_array[50:, 1])
            relation_con_ene = np.corrcoef(features_array[50:, 0], features_array[50:, 2])
            relation_con_hom = np.corrcoef(features_array[50:, 0], features_array[50:, 3])
            relation_cor_ene = np.corrcoef(features_array[50:, 1], features_array[50:, 2])
            relation_cor_hom = np.corrcoef(features_array[50:, 1], features_array[50:, 3])
            relationenehom = np.corrcoef(features_array[50:, 2], features_array[50:, 3])
            relation_list_3 = [(abs(relation_con_cor[0, 1]), 'Contrast&Correlation'), (abs(relation_con_ene[0, 1]), 'Contrast&energy'), (abs(relation_con_hom[0, 1]), 'Contrast&homogeneity'),
                             (abs(relation_cor_ene[0, 1]), 'Correlation&energy'), (abs(relation_cor_hom[0, 1]), 'Correlation&homogeneity'), (abs(relationenehom[0, 1]), 'energy&homogeneity')]
            relation_list_3.sort()

    greatest_relation_1 = relation_list_1[5]
    greatest_relation_2 = relation_list_2[5]
    greatest_relation_3 = relation_list_3[5]

    # print("- Greatest Relations -")
    # print("Asphalt: \n - Value: [",greatest_relation_1[0],"]  Relation: [",greatest_relation_1[1],"]")
    # print("Danger: \n - Value: [",greatest_relation_2[0],"]  Relation: [",greatest_relation_2[1],"]")
    # print("Grass: \n - Value: [",greatest_relation_3[0],"]  Relation: [",greatest_relation_3[1],"]")
    # print()

    # # Contraste e Correlação #
    # plt.scatter(features_array[:26, 0], features_array[:26, 1], color='blue')
    # plt.scatter(features_array[26:51, 0], features_array[26:51, 1], color='red')
    # plt.scatter(features_array[51:, 0], features_array[51:, 1], color='green')
    # plt.xlabel('Contrast')
    # plt.ylabel('Correlation')
    # plt.savefig("./CONvsCOR.png")
    # plt.close()

    # # Contraste e energia #
    # plt.scatter(features_array[:26, 0], features_array[:26, 2], color='darkblue')
    # plt.scatter(features_array[26:51, 0], features_array[26:51, 2], color='crimson')
    # plt.scatter(features_array[51:, 0], features_array[51:, 2], color='darkgreen')
    # plt.xlabel('Contrast')
    # plt.ylabel('energy')
    # plt.savefig("./CONvsene.png")
    # plt.close()

    # # Correlação e energia #
    # plt.scatter(features_array[:26, 1], features_array[:26, 2], color='lightblue')
    # plt.scatter(features_array[26:51, 1], features_array[26:51, 2], color='coral')
    # plt.scatter(features_array[51:, 1], features_array[51:, 2], color='lightgreen')
    # plt.xlabel('Correlation')
    # plt.ylabel('energy')
    # plt.savefig("./CORvsene.png")
    # plt.close()