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

def treinar_neuronio_simples(x, y, w_inicial, taxa_aprendizado, epochs):
    w = w_inicial
    for epoca in range(epochs):
        z = x * w                     # produto escalar
        y_pred = sigmoid(z)           # forward pass
        e = erro(y_pred, y)           # cálculo do erro
        grad = gradiente_perda_peso(y_pred, y, x)  # derivada da perda em relação ao peso
        w = w - taxa_aprendizado * grad            # atualização do peso

        print(f"Época {epoca+1}: y_pred = {y_pred:.4f}, erro = {e:.4f}, peso = {w:.4f}")

treinar_neuronio_simples(
    x=1.0,
    y=0.0,
    w_inicial=0.5,
    taxa_aprendizado=0.1,
    epochs=10
)

