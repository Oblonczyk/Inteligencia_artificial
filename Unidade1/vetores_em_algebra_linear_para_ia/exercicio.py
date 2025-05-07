# IMPLEMENTANDO UMA CAMADA DENSA SIMPLES (SEM FRAMEWORKS)

# Objetivo :  Simular o funcionamento de uma camada densa (fully connected layer), composta por vários neurônios, usando apenas numpy

# Enunciado : implemente a função:
import numpy as np

def camada_densa(entradas: np.ndarray, pesos: np.ndarray, vieses: np.ndarray) -> np.ndarray:
    saidas = []
    for peso, vies in zip(pesos, vieses):
        soma = 0
        for i in range(len(entradas)):
            soma += entradas[i] * peso[i]
        soma += vies
        saidas.append(soma)
    return np.array(saidas)

def testar_camada_densa():
    entradas = np.array([1.0, 2.0, 3.0])
    pesos = np.array([
        [0.2, 0.8, -0.5],
        [0.5, -0.91, 0.26],
        [-0.26, -0.27, 0.17]
    ])
    vieses = np.array([2.0, 3.0, 0.5])

    esperado = np.array([2.3, 2.46, 0.21])
    saida = camada_densa(entradas, pesos, vieses)

    assert np.allclose(saida, esperado, atol=1e-5), f"Esperado {esperado}, mas retornado {saida}"
    print("Teste 1 passou!")

    # Teste com múltiplas entradas (batch)
    entradas_batch = np.array([
        [1.0, 2.0, 3.0],
        [0.5, -0.2, 1.5]
    ])
    esperado_batch = np.array([
        [2.3, 2.46, 0.21],
        [1.19, 3.822, 0.679]
    ])
    saida_batch = np.array([camada_densa(i, pesos, vieses) for i in entradas_batch])

    assert np.allclose(saida_batch, esperado_batch, atol=1e-5), f"Esperado {esperado_batch}, mas retornado {saida_batch}"
    print("Teste 2 passou!")

testar_camada_densa()