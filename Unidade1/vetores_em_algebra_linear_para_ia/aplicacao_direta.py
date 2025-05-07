import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entradas
x = np.array([0.6, 0.8])

# Pesos
w = np.array([0.5, -0.3])

# Viés
b = 0.1

# Cálculo do neurônio
z = np.dot(w, x) + b
y = sigmoid(z)

print(f"Saída do neurônio: {y:.4f}")