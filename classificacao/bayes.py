import numpy as np

""" Este modelo tem duas variáveis por classe: a média e covariância. 

    O método train() pega uma lista de feature arrays (uma por classe) e computa
    a média e covariância para cada um. O método classify() computa 
    as probabilidades da classe para um array de data points e seleciona a classe
    com a maior probabilidade. As classes 

    As class labels estimadas e as probabilidades são retornadas.

    Vantagem: Uma vez que o modelo aprende, não é necessário guardar training data,
                apenas os parâmetros são necessários.
 """

class BayesClassifier(object):

    def __init__(self):
        """ Inicializa o classificador com os dados de treinamento """

        self.labels = [] # class labels
        self.mean = [] # class mean
        self.var = [] # class variances
        self.n = 0 # nbr of classes


    def train(self,data,labels=None):
        """ Treinamento dos dados (list de arrays n*dim)
            Labels são opcionais, padrão é 0...n-1. """
    
        if labels==None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)

        for c in data:
            self.mean.append(np.mean(c,axis=0))
            self.var.append(np.var(c,axis=0))

    def classify(self,points):
        """ Classifica os pontos computando as probabilidades
            para cada classe e retornando o label mais provável """

        # Computa as probabilidades para cada classe
        est_prob = np.array([gauss(m,v,points) for m,v in zip(self.mean,self.var)])
        
        # Pega o índice com maior probabilidade, isso nos dá a class label
        ndx = est_prob.argmax(axis=0)
        est_labels = np.array([self.labels[n] for n in ndx])

        return est_labels, est_prob

def gauss(m,v,x):
    """ Avalia Gaussiana em d dimensões com média independente m e variância v, 
        nos pontos em (linhas) x """

    if len(x.shape)==1:
        n,d = 1,x.shape[0]
    else:
        n,d = x.shape

    # matriz de covariância, subtrair média
    S = np.diag(1/v)
    x = x - m
    # produto das probabilidades
    y = np.exp(-0.5*np.diag(np.dot(x,np.dot(S,x.T))))

    # normaliza e retorna
    return y * (2*np.pi)**(-d/2.0) / (np.sqrt(np.prod(v)) + 1e-6)