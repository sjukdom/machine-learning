import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, input_size, neurons, weights):
        if not weights:
            self.W = np.random.randn(neurons, input_size)
            self.b = np.random.randn(neurons)
        else
            self.W = np.concatenate()

    def preactivation(self, W, x):
        return np.dot()

    def fit(self, x, y, lr, epochs):
        self.W = np.concatenate((self.W, np.ones((self.W.shape[0], 1))), axis=1) # Agregando el bias a la matriz de pesos
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)        # Agregando un 1 (bias) a cada vector de entrada
        errors = np.ones(len(y))                                         # Errores en cada neurona
        for epoch in range(epochs):
            for i, (a, t) in enumerate(zip(x, y)):
                n = np.dot(self.W, a)               # Producto punto de la matriz de pesos y el vector de entrada
                z = self.step(n)                    # Funcion de activacion de la neurona
                e = z - t                           # Calcular el error
                errors[i] = e                       # Guardar error de la neurona i-esima
                self.W = self.W - lr*e*a            # Actualizar bias y los pesos
            if np.sum(np.abs(errors)) == 0:         # Termina si la suma de abs(errores) es 0
                print("Solucion encontrada en la iteracion = ", epoch)
                break

    def preditc(self, x):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        y = np.zeros((x.shape[0]))
        for i, a in enumerate(x):
            n = np.dot(self.W, a)
            z = self.step(n)
            y[i] = z
        return y

    def step(self, x):
        return 1 if x > 1 else 0

# Datos de entrenamiento
x = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])

y = np.array([0,0,0,1])

# Parametros
lr = 0.3
epochs = 20

# Perceptron
p = Perceptron(2, 1)
p.fit(x, y, lr, epochs)
y_pred = p.preditc(x)
print('Prediccion => ', y_pred)

# Parametros de la red neuronal
bias = p.W[:, 2]
weights = p.W[:, 0:2]
print('Bias => ', bias)
print('Pesos =>')
print(weights)
print('Everything')
print(p.W)

# Definir la recta de decision
m = weights[0,0]/weights[0,1]
b = bias/weights[0,1]
t = np.linspace(0, 1.5)
yt = -(m*t + b)

# Visualizacion de los Datos
plt.figure()
plt.title('Perceptron AND')
plt.scatter(x[:3, 0], x[:3, 1])
plt.scatter(x[3, 1], x[3, 1])
plt.plot(t, yt)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
