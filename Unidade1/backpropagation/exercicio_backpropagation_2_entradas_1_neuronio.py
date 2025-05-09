# Backpropagation de 1 neurônio

# Enunciado: Implemente uma rede com 1 neurônio, 1 entrada, com função de ativação sigmoid, e faça:

# 1- Forward pass

# 2- Cálculo do erro com MSE (erro quadrático médio)

# 3- Cálculo da derivada da função sigmoid

# 4- Ajuste do peso com base no gradiente

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def erro(y, r):
    return 1/2 *(y - r) ** 2

def gradiente_perda_peso(y_pred, y_real, x):
    return (y_pred - y_real) * y_pred * (1 - y_pred) * x 

def treinar_neuronio_duplo(x1, x2, y, w1, w2, b, taxa_aprendizado, epochs):
    for epoca in range(epochs):
        z = (x1 * w1) + (x2 * w2) + b                     # produto escalar
        y_pred = sigmoid(z)           # forward pass
        e = erro(y_pred, y)      # cálculo do erro
        grad1 = gradiente_perda_peso(y_pred, y, x1)  # derivada da perda em relação ao peso da entrada 1
        grad2 = gradiente_perda_peso(y_pred, y, x2)  # derivada da perda em relação ao peso da entrada 2
        gradb = (y_pred - y) * y_pred * (1 - y_pred) # vies

        w1 = w1 - taxa_aprendizado * grad1            # atualização do peso 1
        w2 = w2 - taxa_aprendizado * grad2            # atualização do peso 2
        b = b - taxa_aprendizado * gradb              # atualização do viés


        print(f"Época {epoca+1}: y_pred = {y_pred:.4f}, erro = {e:.4f}, peso1 = {w1:.4f}, peso2 = {w2:.4f}", )

treinar_neuronio_duplo(
    x1 = 1.0,
    x2 = 0.5,
    y = 1.0,
    w1 = 0.4,
    w2 = -0.6,
    b = 0.2,
    taxa_aprendizado = 0.1,
    epochs = 10
)

