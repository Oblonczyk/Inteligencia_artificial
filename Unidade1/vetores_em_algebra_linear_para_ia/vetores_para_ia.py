import numpy as np

# Vetores como array NumPy
a = np.array([2, 3])
b = np.array([4, 1])

# Adição
soma = a + b # Realiza a soma de a0 + b0, a1 + b1, ...

# Multiplicação por escalar
esc = 3 * a # realiza a multiplicação de 3*a0, 3*a1, ...

# Produto escalar
dot = np.dot(a, b) # a0*b0 + a1*b1 + ... = s0 + s1 = x

print(f"Soma: {soma}, Escalar: {esc}, Dot: {dot}")
