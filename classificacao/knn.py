import numpy as np

class KnnClassifier(object):
    def __init__(self,labels,samples):
        """ Inicializa o classificador com os dados de treinamento. """

        self.labels = labels
        self.samples = samples

    def classify(self, point, k = 3):
        """ Classifica um ponto de acordo com os k mais próximos nos dados de treinamento.
            Retorna o label. """
        
        # computa a distância para todos os pontos de treinamento
        dist = np.array([L2dist(point,s) for s in self.samples])

        ndx = dist.argsort()

        # uso dicionário para salvar os k mais pertos/vizinhos
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        
        return max(votes)

def L2dist(p1, p2):
    return np.sqrt( sum( (p1-p2)**2 ) )