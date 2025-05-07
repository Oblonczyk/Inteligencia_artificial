# SIMULANDO UMA REDE NEURAL COM TRES CAMADAS DENSAS (SEM FRAMEWORK)

# Objetivo: Fixar o entendimento sobre camadas densas, múltiplos neurônios, função de ativação e o fluxo de dados entre as camadas.

# Enunciado : implemente a função:
import numpy as np

def relu(x):
    return np.maximum(0, x) # Compara cada valor de X com 0, se for menor retorna 0

def camada_densa(entradas: np.ndarray, pesos: np.ndarray, vieses: np.ndarray) -> np.ndarray:
    saidas = []
    for peso, vies in zip(pesos, vieses):
        soma = 0
        for i in range(len(entradas)):
            soma += entradas[i] * peso[i]
        soma += vies
        saidas.append(soma)
    return np.array(saidas)

def rede_tres_camadas(entrada: np.ndarray,
                      pesos1: np.ndarray, vieses1: np.ndarray,
                      pesos2: np.ndarray, vieses2: np.ndarray,
                      pesos3: np.ndarray, vieses3: np.ndarray) -> np.ndarray:
    
    oculta1 = relu(camada_densa(entrada, pesos1, vieses1))
    oculta2 = relu(camada_densa(oculta1, pesos2, vieses2))
    saida = camada_densa(oculta2, pesos3, vieses3)

    return saida

def testar_rede_tres_camadas():
    entrada = np.array([1.0, 2.0, 3.0, 4.0])

    pesos1 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [-0.5, -0.4, -0.3, -0.2],
        [0.3, 0.2, 0.1, 0.0]
    ])
    vieses1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    pesos2 = np.array([
        [0.2, 0.3, 0.5, 0.7, 0.1],
        [0.6, 0.4, 0.3, 0.2, 0.9],
        [0.5, 0.2, 0.6, 0.3, 0.1]
    ])
    vieses2 = np.array([0.5, 1.0, -0.5])

    pesos3 = np.array([
        [0.2, 0.8, -0.1],
        [-0.5, 0.1, 0.6]
    ])
    vieses3 = np.array([1.0, -1.0])

    resultado = rede_tres_camadas(entrada, pesos1, vieses1, pesos2, vieses2, pesos3, vieses3)
    print("Saída final:", resultado)

testar_rede_tres_camadas()