import numpy as np

# Função de ativação sigmoid - transforma um valor em algo entre 0 e 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivada da sigmoid, usada na backpropagation para calcular gradientes
def sigmoid_derivada(a):
    return a * (1 - a)

# Função de treinamento da rede com camada oculta com 2 neurônios e 1 de saída
def treinar_rede_sem_sigmoid_saida(x1, x2, y, w, b, taxa_aprendizado, epochs):
    for epoca in range(epochs):
        ### ===== FORWARD PASS =====

        # Neurônio oculto 1 (N1) - soma ponderada das entradas + viés
        z1 = x1 * w['w11'] + x2 * w['w12'] + b['b1']
        a1 = sigmoid(z1)  # ativação com sigmoid

        # Neurônio oculto 2 (N2)
        z2 = x1 * w['w21'] + x2 * w['w22'] + b['b2']
        a2 = sigmoid(z2)

        # Neurônio de saída (N3) - recebe ativações de N1 e N2
        z3 = a1 * w['w31'] + a2 * w['w32'] + b['b3']
        y_pred = z3

        ### ===== CÁLCULO DO ERRO =====
        erro = 0.5 * (y - y_pred) ** 2  # erro quadrático médio (MSE)

        ### ===== BACKPROPAGATION - CAMADA DE SAÍDA =====
        # Saída (camada 2)
        delta3 = (y_pred - y)
        dw31 = delta3 * a1
        dw32 = delta3 * a2
        db3 = delta3
        delta3 = (y_pred - y)

        ### ===== BACKPROPAGATION - CAMADA OCULTA =====
        # erro local dos neurônios ocultos, propagado da saída
        delta1 = delta3 * w['w31'] * a1 * (1 - a1)
        delta2 = delta3 * w['w32'] * a2 * (1 - a2)


        # gradientes dos pesos da entrada -> neurônios ocultos
        dw11 = delta1 * x1
        dw12 = delta1 * x2
        db1 = delta1

        dw21 = delta2 * x1
        dw22 = delta2 * x2
        db2 = delta2

        ### ===== ATUALIZAÇÃO DOS PESOS E VIESES =====
        # Saída
        w['w31'] -= taxa_aprendizado * dw31
        w['w32'] -= taxa_aprendizado * dw32
        b['b3']  -= taxa_aprendizado * db3

        # Oculta
        w['w11'] -= taxa_aprendizado * dw11
        w['w12'] -= taxa_aprendizado * dw12
        b['b1']  -= taxa_aprendizado * db1

        w['w21'] -= taxa_aprendizado * dw21
        w['w22'] -= taxa_aprendizado * dw22
        b['b2']  -= taxa_aprendizado * db2

        ### ===== PRINT =====
        print(f"Época {epoca+1}: y_pred = {y_pred:.4f}, erro = {erro:.6f}, w11 = {w['w11']:.4f}, w31 = {w['w31']:.4f}")

w = {
    'w11': 0.2, 'w12': -0.4,
    'w21': 0.7, 'w22': 0.1,
    'w31': -0.3, 'w32': 0.8
}

b = {
    'b1': 0.0,
    'b2': 0.0,
    'b3': 0.0
}

treinar_rede_sem_sigmoid_saida(
    x1=1.0,
    x2=0.5,
    y=2.0,
    w=w,
    b=b,
    taxa_aprendizado=0.05,
    epochs=15
)