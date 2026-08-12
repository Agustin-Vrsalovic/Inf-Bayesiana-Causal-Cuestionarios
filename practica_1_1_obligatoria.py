"""
Práctica 1.1: Argumentos causales alternativos e incertidumbre.
============================================
"""

import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Definición del espacio de hipótesis
# ------------------------------------------------------------

# H representa las 3 cajas posibles {0, 1, 2}
H = np.arange(3)


# ------------------------------------------------------------
# 2. Priors
# ------------------------------------------------------------


def p_r(r: int) -> float:
    """
    P(R = r).
    Prior sobre la ubicación del regalo.
    r ∈ {0, 1, 2}
    """
    return 1/3


def p_c(c: int) -> float:
    """
    P(C = c).
    Prior sobre la caja elegida por el participante
    c ∈ {0, 1, 2}
    """
    return 1/3


# ------------------------------------------------------------
# 3. Modelos del presentador
# ------------------------------------------------------------


def p_s_rM0(s: int, r: int) -> float:
    """
    P(S = s | R= r, M = 0).
    Modelo 0 (No Monty Hall):
    El presentador abre cualquier caja que no tenga el regalo.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    """
    if s == r:
        return 0
    else:
        return 1/2


def p_s_rcM1(s: int, r: int, c: int):
    """
    P(S = s | R= r, C= c, M = 1).
    Modelo 1 (Monty Hall):
    El presentador abre una caja que no tenga el regalo
    ni haya sido seleccionada.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    """
    if s == r:
        return 0
    elif s == c:
        return 0
    elif r == c:
        return 1/2
    else:
        return 1


# ------------------------------------------------------------
# 4. Distribución conjunta P(r, c, s | M)
# ------------------------------------------------------------


def p_rcs_M(r: int, c: int, s: int, m: int) -> float:
    """
    P(r, c, s | M) = P(r | M)P(c | M)P(s | r, c, M)
    Distribución conjunta del modelo m.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    m ∈ {0, 1}
    """
    if m == 0:
        return p_r(r) * p_c(c) * p_s_rM0(s=s, r=r)
    else:
        return p_r(r) * p_c(c) * p_s_rcM1(s=s, r=r, c=c)


# ------------------------------------------------------------
# 5. Simulación de datos (asumiendo Monty Hall verdadero)
# ------------------------------------------------------------

np.random.seed(1)


def simular(T=16) -> List[Tuple[int, int, int]]:
    """
    Función para simular datos según el modelo Monty Hall verdadero.
    T: número de datos a generar.
    """
    episodios = []
    for episodio in range(T):
        r = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
        c = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
        s = np.random.choice([0, 1, 2], p=[p_s_rcM1(s=0, r=r, c=c),
                                           p_s_rcM1(s=1, r=r, c=c),
                                           p_s_rcM1(s=2, r=r, c=c)])
        episodios.append((c, s, r))
    return episodios


# ------------------------------------------------------------
# 6. Probabilidad de los datos dado un modelo
# ------------------------------------------------------------



def pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    P(Datos | M) = prod([ P(c,s,r|M) for c, s, r in Datos ])
    Probabilidad de ver los datos dados el modelo considerado.
    m ∈ {0, 1}
    """
    productoria = 1
    for episodio in range(len(datos)):
        productoria *= p_rcs_M(c=datos[episodio][0], s=datos[episodio][1], r=datos[episodio][2], m=m)
    
    return productoria

# ------------------------------------------------------------
# 7. Probabilidad total de ver los datos
# ------------------------------------------------------------


def pM(m: int) -> float:
    """
    P(M)
    Prior sobre los modelos.
    m ∈ {0, 1}
    """
    return 1/2


def pDatos(datos: List[Tuple[int, int, int]]) -> float:
    """
    P(Datos)
    Probabilidad total de ver los datos usando la contribución
    de todos los modelos.
    """
    return pDatos_M(datos=datos, m=0) * pM(0) + pDatos_M(datos=datos, m=1) * pM(1)


# ------------------------------------------------------------
# 8. Actualización de creencias sobre los modelos
# ------------------------------------------------------------


def pM_Datos(m: int, datos: List[Tuple[int, int, int]]) -> float:
    """
    P(M | Datos)
    Actualización de la creencia sobre el modelo M dado los datos
    (distribución a posteriori).
    """
    return (pDatos_M(datos=datos, m=m) * pM(m)) / pDatos(datos = datos)


def evolucion_posterior(m: int, datos: List[Tuple[int, int, int]]) -> List[float]:
    """
    Evolución del posterior de los modelos según se observa información.
    """
    posteriores= []

    for episodio in range(1, len(datos) + 1):
        datos_observados = datos[:episodio]
        posteriores.append(pM_Datos(m=m, datos=datos_observados))

    return posteriores



# ------------------------------------------------------------
# 9. Cálculo del bayes factor
# (diferencia en órdenes de magnitud en escala logarítmica)
# ------------------------------------------------------------


def log_pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log(P(Datos | M))
    Logaritmo de la probabilidad de ver los datos dados el modelo considerado.
    """
    return np.log10(pDatos_M(datos, m))


def log_bayes_factor(datos: List[Tuple[int, int, int]], m_i: int, m_j: int) -> float:
    """
    log(Datos | Mi / Datos | Mj) = log_10P(Datos | Mi) - log_10P(Datos | Mj)
    Bayes factor en escala logarítmica para comparar dos modelos.
    m_i, m_j ∈ {0, 1}
    """
    return log_pDatos_M(datos, m_i) - log_pDatos_M(datos, m_j)


# ------------------------------------------------------------
# 10. Predicción típica que realizan los modelos sobre los datos
# ------------------------------------------------------------

def prediccion_tipica(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log(Media Geométrica)
    Logaritmo de la media geométrica de los datos dados el modelo considerado.
    m ∈ {0, 1}
    """
    return log_pDatos_M(datos, m) / len(datos)



# ------------------------------------------------------------
# 11. Ejecución principal
# ------------------------------------------------------------

if __name__ == "__main__":
    # Simulación
    datos = simular(T=16)

    # Print de funciones
    print("log_bayes_factor (M1 contra M0):", log_bayes_factor(datos, 1, 0))
    print("prediccion_tipica (M0):", prediccion_tipica(datos, 0))
    print("prediccion_tipica (M1):", prediccion_tipica(datos, 1))

    # Posteriores
    post_M0 = evolucion_posterior(0, datos)
    post_M1 = evolucion_posterior(1, datos)

    # Gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(post_M0, label="M0: Base")
    plt.plot(post_M1, label="M1: Monty Hall")
    plt.xlabel("Número de episodios")
    plt.ylabel("P(Modelo | Datos)")
    plt.title("Evolución del posterior de los modelos")
    plt.legend()
    plt.tight_layout()
    plt.show()
