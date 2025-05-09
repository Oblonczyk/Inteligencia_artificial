import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivada(a):
    return a * (1 - a)  # derivada de sigmoid em termos da própria saída

def treinar_rede_oculta_dupla(x1, x2, y, w, b, taxa_aprendizado, epochs):
    for epoca in range(epochs):
        ### ===== FORWARD PASS =====

        # Neurônio oculto 1 (N1)
        z1 = x1 * w['w11'] + x2 * w['w12'] + b['b1']
        a1 = sigmoid(z1)

        # Neurônio oculto 2 (N2)
        z2 = x1 * w['w21'] + x2 * w['w22'] + b['b2']
        a2 = sigmoid(z2)

        # Neurônio de saída (N3)
        z3 = a1 * w['w31'] + a2 * w['w32'] + b['b3']
        y_pred = sigmoid(z3)

        ### ===== CÁLCULO DO ERRO =====
        erro = 0.5 * (y - y_pred) ** 2

        ### ===== BACKPROPAGATION =====

        # Saída (camada 2)
        delta3 = (y_pred - y) * sigmoid_derivada(y_pred)
        dw31 = delta3 * a1
        dw32 = delta3 * a2
        db3 = delta3

        # Oculta (camada 1)
        delta1 = delta3 * w['w31'] * sigmoid_derivada(a1)
        delta2 = delta3 * w['w32'] * sigmoid_derivada(a2)

        dw11 = delta1 * x1
        dw12 = delta1 * x2
        db1 = delta1

        dw21 = delta2 * x1
        dw22 = delta2 * x2
        db2 = delta2

        ### ===== ATUALIZAÇÃO DOS PESOS =====
        # Camada de saída
        w['w31'] -= taxa_aprendizado * dw31
        w['w32'] -= taxa_aprendizado * dw32
        b['b3']  -= taxa_aprendizado * db3

        # Camada oculta
        w['w11'] -= taxa_aprendizado * dw11
        w['w12'] -= taxa_aprendizado * dw12
        b['b1']  -= taxa_aprendizado * db1

        w['w21'] -= taxa_aprendizado * dw21
        w['w22'] -= taxa_aprendizado * dw22
        b['b2']  -= taxa_aprendizado * db2

        ### ===== PRINT =====
        print(f"Época {epoca+1}: y_pred = {y_pred:.4f}, erro = {erro:.6f}, w11 = {w['w11']:.4f}, w31 = {w['w31']:.4f}")

w = {
    'w11': 0.1, 'w12': 0.2,
    'w21': -0.1, 'w22': 0.1,
    'w31': 0.4, 'w32': 0.3
}

b = {
    'b1': 0.0,
    'b2': 0.0,
    'b3': 0.0
}

treinar_rede_oculta_dupla(
    x1=1.0,
    x2=0.5,
    y=1.0,
    w=w,
    b=b,
    taxa_aprendizado=0.1,
    epochs=10
)