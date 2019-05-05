import numpy as np
from tqdm import tqdm

class MLPerceptron():
    def __init__(self, layers, activations):
        self.W = [np.random.randn(i, j)*np.sqrt(1.0/(i+j)) for i, j in zip(layers[1:], layers[:-1])]     # Lista de pesos por capa
        self.b = [np.random.randn(n) for n in layers[1:]]                                                # Lista de bias por capa
        self.f = self.getActivations(activations)                                                        # Funciones de activacion
        self.df = self.getDerivatives(activations)                                                       # Derivada de activaciones
        self.layers = layers[1:]                                                                         # Neuronas por capa
        self.L = len(layers) - 1                                                                         # Numero de capas de la red
        self.n = [0]*(len(layers)-1)                                                                     # Lista de preactivaciones
        self.a = [0]*len(layers)                                                                         # Lista de activaciones

    def getActivations(self, activations):
        functions = {
            "sigmoid": self.sigma,
            "purelin": self.purelin,
            "tanh": self.tanh,
            "relu": self.relu
        }
        return [functions[a] for a in activations]

    def getDerivatives(self, activations):
        functions = {
            "sigmoid": self.dxSigma,
            "purelin": self.dxPurelin,
            "tanh": self.dxTanh,
            "relu": self.dxRelu
        }
        return [functions[a] for a in activations]

    def FeedForward(self, x):
        '''
            Calcular la salida de la red neuronal para el vector x
        '''
        ar = x
        self.a[0] = x
        for r in range(self.L):
            nr = np.dot(self.W[r], ar) + self.b[r]
            ar = self.f[r](nr)
            self.n[r] = nr
            self.a[r+1] = ar
        return ar

    def Error(self, t, y):
        return t - y

    def Backpropagation(self, error):
        s = [0]*self.L
        for r in range(self.L-1, -1, -1):
            F = np.diag(self.df[r](self.n[r].reshape(-1)))
            if r == self.L-1:
                sr = -2*np.dot(F, error)
            else:
                sr = np.dot(F, np.dot(self.W[r+1].T, sr))
            s[r] = sr
        return s

    def UpdateWeights(self, s, lr):
        for r in range(self.L-1, -1, -1):
            dW = np.dot(s[r].reshape(-1,1), self.a[r].reshape(-1,1).T)
            dB = s[r]
            self.W[r] -= lr*dW
            self.b[r] -= lr*dB

    def fit(self, x, t, lr, epochs):
        errors = []
        for i in tqdm(range(epochs)):
            for xn, tn in zip(x, t):
                prediction = self.FeedForward(xn)
                error = self.Error(tn, prediction)
                delta = self.Backpropagation(error)
                self.UpdateWeights(delta, lr)
            errors.append(error)
        return errors

    def predict(self, x):
        y = [0]*len(x)
        for i, xn in enumerate(x):
            y[i] = self.FeedForward(xn)[0]
        return y

    def sigma(self, x):
        return 1.0/(1.0+np.exp(-x))

    def dxSigma(self, x):
        return self.sigma(x) * (1.0 - self.sigma(x))

    def purelin(self, x):
        return x

    def dxPurelin(self, x):
        return np.array([1.0])

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def dxTanh(self, x):
        return 1.0 - np.square(self.tanh(x))

    def relu(self, x):
        return x if x>=0 else 0

    def dxRelu(self, x):
        return np.array([1])
