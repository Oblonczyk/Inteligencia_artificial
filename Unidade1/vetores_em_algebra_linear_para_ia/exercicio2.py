# SIMULANDO UMA REDE NEURAL COM DUAS CAMADAS DENSAS (SEM FRAMEWORK)

# Objetivo: Simular duas camadas densas conectadas (camada oculta + camada de saída), fazendo o forward pass completo de uma mini-rede neural manualmente.

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

def rede_duas_camadas(entrada: np.ndarray,
                      pesos1: np.ndarray, vieses1: np.ndarray,
                      pesos2: np.ndarray, vieses2: np.ndarray) -> np.ndarray:
    
    oculta = relu(camada_densa(entrada, pesos1, vieses1))
    saida = camada_densa(oculta, pesos2, vieses2)

    return saida
    
def testar_rede_duas_camadas():
    entrada = np.array([1.0, 2.0, 3.0])
    pesos1 = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [-0.5, -0.6, -0.7]
    ])
    vieses1 = np.array([1.0, 1.5, 2.0])

    pesos2 = np.array([
        [0.7, 0.8, 0.9],
        [1.0, -1.0, 0.5]
    ])
    vieses2 = np.array([0.5, -1.0])

    esperado = np.array([5.94, -3.3])
    resultado = rede_duas_camadas(entrada, pesos1, vieses1, pesos2, vieses2)
    print(resultado)

    assert np.allclose(resultado, esperado, atol=1e-2), f"Esperado {esperado}, mas retornado {resultado}"
    print("Teste passou!")

testar_rede_duas_camadas()