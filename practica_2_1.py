"""
Práctica 2.1: Métodos de inferencia exacta.
===========================================
"""

import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt

import ModeloLineal as ml
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import OLS

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Semilla para reproducibilidad
np.random.seed(1)

# Constantes del problema
N: int = 20  # Cantidad de datos
D: int = 10  # Cantidad de modelos (grados 0 a 9)
BETA: float = (1 / 0.2) ** 2  # Precisión de los datos (inverso de varianza)
ALPHA: float = 10e-6  # Precisión del prior (inverso de varianza del prior)


# =============================================================================
# SECCIÓN 1: GENERACIÓN DE DATOS
# =============================================================================


def realidad_causal_subyacente(X: np.ndarray, beta: float = BETA) -> np.ndarray:
    """
    Genera datos de una función seno con ruido gaussiano.

    La función objetivo es: y = sin(2*pi*x) + ruido
    donde ruido ~ N(0, 1/beta)

    Args:
        X: Array de valores de entrada con shape (N, 1)
        beta: Precisión del ruido (inverso de varianza)

    Returns:
        Array con valores de salida ruidosos, mismo shape que X
    """
    ruido = np.random.normal(
        loc=0,
        scale=1 / np.sqrt(beta),
        size=X.shape
    )
    Y = np.sin(2 * np.pi * X) + ruido

    return Y


def generar_datos() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera datos de entrenamiento y grilla de evaluación.

    Returns:
        X: Datos de entrada de entrenamiento, shape (N, 1)
        Y: Datos de salida de entrenamiento, shape (N, 1)
        X_grilla: Puntos para graficar la función suave, shape (100, 1)
        Y_grilla: Función verdadera en grilla sin ruido, shape (100, 1)
    """
    X = np.random.uniform(-0.5, 0.5, size=(N, 1))
    Y = realidad_causal_subyacente(X, beta=BETA)

    X_grilla = np.linspace(-0.5, 0.5, 100).reshape(-1, 1)
    Y_grilla = np.sin(2 * np.pi * X_grilla)

    return X, Y, X_grilla, Y_grilla


"""
import matplotlib.pyplot as plt

plt.scatter(X, Y, label="Datos")
plt.plot(X_grilla, Y_grilla, label="Seno")
plt.ylim(-1.5, 1.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
"""

# =============================================================================
# SECCIÓN 2: MÁXIMA VEROSIMILITUD (OLS)
# =============================================================================


def phi(X: np.ndarray, grado: int) -> pd.DataFrame:
    """
    Genera la matriz de diseño polinomial (feature matrix).

    Transforma X en una matriz con columnas [1, X, X^2, ..., X^grado].
    Esta transformación convierte la regresión polinomial no lineal
    en un problema de regresión lineal en los parámetros.

    Ejemplo: Si X = [2] y grado = 2, retorna [[1, 2, 4]]

    Args:
        X: Array de entrada con shape (N, 1)
        grado: Grado máximo del polinomio

    Returns:
        DataFrame con (grado+1) columnas: [X^0, X^1, ..., X^grado]
        Tiene shape (N, grado+1)
    """
    columnas = {}

    for d in range(grado + 1):
        columnas[f"X^{d}"] = X[:, 0] ** d

    return pd.DataFrame(columnas)


def ajustar_modelos_OLS(X: np.ndarray, Y: np.ndarray) -> List:
    """
    Ajusta modelos polinomiales de grado 0 a D-1 por OLS (Ordinary Least Squares).

    OLS minimiza: suma de (Y_predicho - Y_observado)^2

    Args:
        X: Datos de entrada, shape (N, 1)
        Y: Datos de salida, shape (N, 1)

    Returns:
        Lista de D modelos OLS ajustados (uno por cada grado)
    """
    modelos = []

    for grado in range(D):
        X_transformado = phi(X, grado)

        modelo = OLS(Y, X_transformado)
        modelo_ajustado = modelo.fit()

        modelos.append(modelo_ajustado)

    return modelos


def extraer_verosimilitudes_OLS(modelos: List) -> np.ndarray:
    """
    Extrae las log-verosimilitudes de los modelos OLS ajustados.

    La verosimilitud mide cuán probable son los datos observados bajo
    el modelo ajustado. Mayor verosimilitud = mejor ajuste a los datos.

    Args:
        modelos: Lista de modelos OLS ajustados

    Returns:
        Array de log-verosimilitudes, shape (D,)
        El atributo .llf es "log likelihood function"
    """
    log_likelihoods = []

    for modelo in modelos:
        log_likelihoods.append(modelo.llf)

    return np.array(log_likelihoods)


"""
#Prueba de lo que hice hasta ahora:
X, Y, X_grilla, Y_grilla = generar_datos()
modelos = ajustar_modelos_OLS(X, Y)

#Calculo y Observacion de las curvas OLS
def obtener_curvas_OLS(modelos: List, X_grilla: np.ndarray) -> List:
    curvas = []

    for modelo in modelos:
        grado = len(modelo.params) - 1

        X_transformado = phi(X_grilla, grado)

        Y_pred = X_transformado @ modelo.params

        curvas.append(Y_pred)

    return curvas

curvas = obtener_curvas_OLS(modelos, X_grilla)

#Grafico
for grado, curva in enumerate(curvas):
    plt.plot(X_grilla, curva, label=f"Grado {grado}")

plt.scatter(X, Y)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
"""

#1.5 El Balance Natural:

# =============================================================================
# SECCIÓN 3: PREDICCIÓN ONLINE
# =============================================================================


def prediccion_puntual(y: float, x: float, h: np.ndarray) -> float:
    """
    Calcula P(y | x, h, beta) = densidad de probabilidad gaussiana.

    Dado un valor observado y, la posición x, y coeficientes h del modelo,
    calcula cuán probable es observar y bajo el modelo gaussiano.

    La predicción es: y ~ N(mu, sigma^2)
    donde mu = sum(h_i * x^i) y sigma^2 = 1/beta

    Args:
        y: Valor observado (escalar)
        x: Posición de entrada (escalar)
        h: Coeficientes del modelo [h_0, h_1, h_2, ...]

    Returns:
        Densidad de probabilidad en y (escalar positivo)
    """
    mu = sum(h[i] * x**i for i in range(len(h)))
    sigma = 1 / np.sqrt(BETA)
    
    densidad = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(        -((y - mu)**2) / (2 * sigma**2)    )
    
    return densidad


def evaluacion_online_OLS(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Evalúa modelos secuencialmente (online): predice cada dato
    con modelo entrenado hasta ese momento, luego readjusta.

    Este es un procedimiento de validación cruzada secuencial.
    Mide capacidad predictiva real, no overfitting en entrenamiento.

    Args:
        X: Datos de entrada, shape (N, 1)
        Y: Datos de salida, shape (N, 1)

    Returns:
        Array de log-evidencias acumuladas, shape (D,)
        Modelos con mejor predicción online tienen valores más altos
    """
    log_evidencias_acumulada = [0 for _ in range(D)]

    modelos = []
    for grado in range(D):
        modelo = OLS(Y[:1], phi(X[:1], grado)).fit()
        for i in range(1, N):
            pred_y = prediccion_puntual(Y[i], X[i], modelo.params)
            log_evidencias_acumulada[grado-1] += float(np.log(pred_y))
            modelo = OLS(Y[:i+1], phi(X[:i+1], grado)).fit()
    return log_evidencias_acumulada


# =============================================================================
# SECCIÓN 4: EVIDENCIA BAYESIANA Y SELECCIÓN DE MODELOS
# =============================================================================


def calcular_log_evidencias(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calcula la evidencia Bayesiana p(datos | modelo) para cada grado.

    La evidencia es la verosimilitud marginal: integra sobre todos
    los valores posibles de los parámetros según el prior.

    Incorpora automáticamente un balance entre:
    - Ajuste a los datos (mayor verosimilitud)
    - Complejidad del modelo (penaliza grados de libertad)

    Este es el principio de la "Navaja de Occam Bayesiana".

    Args:
        X: Datos de entrada, shape (N, 1)
        Y: Datos de salida, shape (N, 1)

    Returns:
        Array de log-evidencias, shape (D,)
        Valores más altos indican mejor balance complejidad-ajuste
    """
    log_evidencias = []

    for grado in range(D):
        Phi = phi(X, grado).to_numpy()

        log_ev = ml.log_evidence(Y, Phi, ALPHA, BETA)

        log_evidencias.append(log_ev.item())

    return np.array(log_evidencias)


def calcular_posterior_modelos(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calcula P(modelo | datos) usando la regla de Bayes.

    Fórmula: P(M | D) = P(D | M) * P(M) / P(D)

    Asumiendo priors uniformes P(M) = 1/D para todos los modelos,
    el posterior es proporcional a la evidencia.

    Normaliza para que las probabilidades sumen a 1.

    Args:
        X: Datos de entrada, shape (N, 1)
        Y: Datos de salida, shape (N, 1)

    Returns:
        Array de probabilidades posteriores, shape (D,)
        Suma a 1. Indica credibilidad de cada modelo.
    """
    log_evidencias = calcular_log_evidencias(X=X, Y=Y)

    prior = 1/D
    evidencias = np.exp(log_evidencias)
    posterior_no_normalizado = evidencias * prior

    posterior = posterior_no_normalizado / posterior_no_normalizado.sum()
    return posterior


# =============================================================================
# SECCIÓN 5: DISTRIBUCIÓN PREDICTIVA CON INCERTIDUMBRE
# =============================================================================


def ajustar_modelos_MAP(
    X: np.ndarray, Y: np.ndarray, alpha_factor: float = 1.0
) -> List[dict]:
    """
    Ajusta modelos usando Máximo A Posteriori (MAP).

    MAP añade un prior regulador a los coeficientes, penalizando
    valores grandes. Esto reduce overfitting en comparación con OLS.

    Args:
        X: Datos de entrada, shape (N, 1)
        Y: Datos de salida, shape (N, 1)
        alpha_factor: Factor para escalar la fuerza del prior
                      (mayor = más regularización)

    Returns:
        Lista de diccionarios con:
        - "media": Coeficientes del posterior [h_0, h_1, ...]
        - "covarianza": Matriz de covarianza (incertidumbre en coeficientes)
        - "grado": Grado polinomial
    """
    pass


# =============================================================================
# SECCIÓN 6: MODELOS CAUSALES ALTERNATIVOS (DATOS DE ALTURAS)
# =============================================================================
""" Lectura del DataFrame
import pandas as pd

df_alturas = pd.read_csv("CSVs/alturas.csv")
df_alturas.head()
"""

def construir_modelo_base(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de diseño y vector de respuesta para el modelo base.

    Modelo: altura = h0 + h1 * altura_madre

    Hipótesis: la altura del hijo depende linealmente de la altura de la madre,
    sin considerar el sexo biológico ni la identidad grupal del individuo.

    Args:
        df: DataFrame con columnas 'altura', 'altura_madre', 'sexo'

    Returns:
        PHI: Matriz de diseño, shape (N, 2) con columnas [1, altura_madre]
        Y: Vector de alturas observadas, shape (N, 1)
    """

    alturas = df['altura'].to_numpy().reshape(-1, 1)
    alturas_madre = df['altura_madre'].to_numpy().reshape(-1, 1)

    PHI = phi(X=alturas_madre, grado=1).to_numpy()
    Y = alturas

    return PHI, Y


def construir_modelo_biologico(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de diseño y vector de respuesta para el modelo biológico.

    Modelo: altura = h0 + h1 * altura_madre + h2 * I(sexo=F)

    Hipótesis: el sexo biológico tiene un efecto aditivo sobre la altura,
    controlando por la altura de la madre. El indicador I(sexo=F) vale 1
    para mujeres y 0 para varones.

    Args:
        df: DataFrame con columnas 'altura', 'altura_madre', 'sexo'

    Returns:
        PHI: Matriz de diseño, shape (N, 3) con columnas [1, altura_madre, I(sexo=F)]
        Y: Vector de alturas observadas, shape (N, 1)
    """
    alturas = df['altura'].to_numpy().reshape(-1, 1)
    alturas_madre = df['altura_madre'].to_numpy().reshape(-1, 1)
    sexos = df['sexo'].to_numpy().reshape(-1, 1)
    indicador_sexo = (sexos == 'F').astype(int)

    PHI = phi(X=alturas_madre, grado=1).to_numpy()
    PHI = np.hstack((PHI, indicador_sexo)) #Agrego la segunda variable

    Y = alturas

    return PHI, Y


def construir_modelo_identitario(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de diseño para el modelo identitario (grupos de dos personas).

    Modelo: una ordenada propia para cada grupo de 2 personas + altura_madre.

    Hipótesis: existen agrupaciones de dos personas con similitudes intragrupales
    que pueden confundir la estimación del efecto del sexo. Cada grupo recibe
    su propio intercepto mediante una variable indicadora.

    Args:
        df: DataFrame con columnas 'altura', 'altura_madre', 'sexo'

    Returns:
        PHI: Matriz de diseño, shape (N, 1 + N//2) con columnas
             [altura_madre, I_grupo_0, I_grupo_1, ..., I_grupo_{N//2 - 1}]
        Y: Vector de alturas observadas, shape (N, 1)
    """
    alturas = df['altura'].to_numpy().reshape(-1, 1)
    alturas_madre = df['altura_madre'].to_numpy().reshape(-1, 1)

    cant_grupos = len(df)//2
    grupos = np.repeat(np.arange(cant_grupos), 2) #Ej: [0,0,1,1,2,2]
    indicadores_grupo = np.eye(cant_grupos)[grupos]
    PHI = np.hstack((alturas_madre, indicadores_grupo)) #No usamos phi() porque no queremos un intercept global

    Y = alturas

    return PHI, Y


def comparar_modelos_altura(df: pd.DataFrame) -> dict:
    """
    Compara los tres modelos causales usando evidencia Bayesiana.

    Calcula p(datos | modelo) para los modelos base, biológico e identitario,
    y devuelve el posterior normalizado sobre los tres modelos asumiendo
    priors uniformes.

    Args:
        df: DataFrame con columnas 'altura', 'altura_madre', 'sexo'

    Returns:
        Diccionario con:
        - "evidencia_base": float, log-evidencia del modelo base
        - "evidencia_biologico": float, log-evidencia del modelo biológico
        - "evidencia_identitario": float, log-evidencia del modelo identitario
        - "mejor_modelo": str, nombre del modelo con mayor log-evidencia
        - "posterior_modelos": dict con probabilidades posteriores normalizadas
    """
    prior = 1/3

    Phi_base, Y_base = construir_modelo_base(df)
    Phi_biologico, Y_biologico = construir_modelo_biologico(df=df)
    Phi_identitario, Y_identitario = construir_modelo_identitario(df)
    
    log_ev_base = ml.log_evidence(Y_base, Phi_base, ALPHA, BETA)
    log_ev_biologico = ml.log_evidence(Y_biologico, Phi_biologico, ALPHA, BETA)
    log_ev_identitario = ml.log_evidence(Y_identitario, Phi_identitario, ALPHA, BETA)

    log_evidencias = [
        log_ev_base,
        log_ev_biologico,
        log_ev_identitario
    ]

    nombres = ["base", "biologico", "identitario"]

    indice_mejor = np.argmax(log_evidencias)
    mejor_modelo = nombres[indice_mejor]

    #for i in range(len(log_evidencias)):
        #mejor_modelo = log_evidencias[0]
        #if log_evidencias[i] > mejor_modelo:
        #    mejor_modelo = log_evidencias[i]

    evidencias = np.exp(log_evidencias)
    posterior_no_normalizado = evidencias * prior

    posterior = posterior_no_normalizado / posterior_no_normalizado.sum()

    posterior_modelos = {
        "base": posterior[0],
        "biologico": posterior[1],
        "identitario": posterior[2]
    }

    #return log_evidencias, mejor_modelo, posterior_modelos
    return {
        "evidencia_base": log_ev_base,
        "evidencia_biologico": log_ev_biologico,
        "evidencia_identitario": log_ev_identitario,
        "mejor_modelo": mejor_modelo,
        "posterior_modelos": posterior_modelos
    }


def calcular_media_geometrica(log_evidencia: float, n: int) -> float:
    """
    Calcula la media geométrica de la evidencia por dato.

    Permite comparar evidencias entre datasets de distinto tamaño,
    normalizando por la cantidad de datos n.

    Fórmula: exp(log_evidencia / n)

    Args:
        log_evidencia: Log-evidencia del modelo (escalar)
        n: Número de datos usados para calcular la evidencia

    Returns:
        Media geométrica de la evidencia por dato (escalar positivo)
    """
    return np.exp(log_evidencia / n)


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # =========================================================================
    # FUNCIONES DE VISUALIZACIÓN
    # =========================================================================

    def plotear_datos(
        X: np.ndarray,
        Y: np.ndarray,
        X_grilla: np.ndarray,
        Y_grilla: np.ndarray,
    ) -> None:
        """Grafica la función objetivo y los datos observados."""
        plt.figure(figsize=(8, 6))

        # Graficar la función verdadera como línea discontinua
        plt.plot(
            X_grilla,
            Y_grilla,
            "--",
            color="black",
            label="Función verdadera",
            linewidth=2,
        )

        # Graficar los datos observados como puntos discretos
        plt.scatter(X, Y, color="black", s=50, label="Datos observados", zorder=5)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Función objetivo y datos muestreados")
        plt.ylim(-1.5, 1.5)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plotear_verosimilitudes(log_likelihoods: np.ndarray) -> None:
        """
        Grafica las verosimilitudes de todos los modelos.

        Args:
            log_likelihoods: Array de log-verosimilitudes, shape (D,)
                             Resultado de extraer_verosimilitudes_OLS().
        """
        plt.figure(figsize=(8, 6))

        # Convertir log-verosimilitudes a verosimilitudes (escala lineal)
        verosimilitudes = np.exp(log_likelihoods)

        # Generar colores distintos para cada barra
        colores = plt.cm.tab10(np.linspace(0, 1, D))

        for grado in range(D):
            plt.bar(grado, verosimilitudes[grado], color=colores[grado], alpha=0.8)

        plt.xlabel("Grado del polinomio")
        plt.ylabel("Máxima verosimilitud")
        plt.title("Máxima verosimilitud por modelo")
        plt.xticks(range(D))
        plt.grid(True, alpha=0.3, axis="y")

    def plotear_ajustes_OLS(
        modelos: List,
        X: np.ndarray,
        Y: np.ndarray,
        X_grilla: np.ndarray,
        Y_grilla: np.ndarray,
    ) -> None:
        """
        Grafica todas las curvas ajustadas por OLS superpuestas.

        Args:
            modelos: Lista de modelos OLS ajustados.
                     Resultado de ajustar_modelos_OLS().
            X: Datos de entrada, shape (N, 1)
            Y: Datos de salida, shape (N, 1)
            X_grilla: Grilla para evaluación, shape (100, 1)
            Y_grilla: Función verdadera en grilla, shape (100, 1)
        """
        plt.figure(figsize=(12, 8))

        colores = plt.cm.tab10(np.linspace(0, 1, D))

        plt.plot(
            X_grilla,
            Y_grilla,
            "--",
            color="black",
            label="Función verdadera",
            linewidth=2,
        )
        plt.scatter(X, Y, color="black", s=50, label="Datos", zorder=5)

        for grado, modelo in enumerate(modelos):
            coeficientes = modelo.params.values
            predicciones = np.array(
                [
                    sum(coeficientes[i] * (x**i) for i in range(len(coeficientes)))
                    for x in X_grilla.ravel()
                ]
            )
            plt.plot(
                X_grilla,
                predicciones,
                color=colores[grado],
                label=f"Grado {grado}",
                alpha=0.8,
                linewidth=1.5,
            )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Ajustes OLS por grado polinomial")
        plt.ylim(-1.5, 1.5)
        plt.legend(ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)

    def plotear_mejor_ajuste_OLS(
        modelos: List,
        X: np.ndarray,
        Y: np.ndarray,
        X_grilla: np.ndarray,
        Y_grilla: np.ndarray,
        grado_mejor: int = 9,
    ) -> None:
        """
        Grafica solo el mejor modelo OLS (mayor verosimilitud).

        Args:
            modelos: Lista de modelos OLS ajustados.
                     Resultado de ajustar_modelos_OLS().
            X: Datos de entrada, shape (N, 1)
            Y: Datos de salida, shape (N, 1)
            X_grilla: Grilla para evaluación, shape (100, 1)
            Y_grilla: Función verdadera en grilla, shape (100, 1)
            grado_mejor: Grado del mejor modelo (por defecto 9)
        """
        plt.figure(figsize=(8, 6))

        colores = plt.cm.tab10(np.linspace(0, 1, D))

        plt.plot(
            X_grilla,
            Y_grilla,
            "--",
            color="black",
            label="Función verdadera",
            linewidth=2,
        )
        plt.scatter(X, Y, color="black", s=50, label="Datos", zorder=5)

        modelo_mejor = modelos[grado_mejor]
        coeficientes = modelo_mejor.params.values
        predicciones = np.array(
            [
                sum(coeficientes[i] * (x**i) for i in range(len(coeficientes)))
                for x in X_grilla.ravel()
            ]
        )

        plt.plot(
            X_grilla,
            predicciones,
            color=colores[grado_mejor],
            label=f"Grado {grado_mejor}",
            linewidth=2,
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Mejor ajuste OLS (máxima verosimilitud)")
        plt.ylim(-1.5, 1.5)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plotear_evaluacion_online_OLS(log_evidencias: np.ndarray) -> None:
        """
        Grafica capacidad predictiva secuencial de todos los modelos.

        Args:
            log_evidencias: Array de log-evidencias acumuladas, shape (D,).
                            Resultado de evaluacion_online_OLS().
        """
        evidencias = np.exp(log_evidencias)

        colores = plt.cm.tab10(np.linspace(0, 1, D))

        plt.figure(figsize=(8, 6))

        for grado in range(D):
            plt.bar(grado, evidencias[grado], color=colores[grado], alpha=0.8)

        plt.xlabel("Grado del polinomio")
        plt.ylabel("Evidencia acumulada (predicción online)")
        plt.title("Capacidad predictiva secuencial (OLS)")
        plt.xticks(range(D))
        plt.grid(True, alpha=0.3, axis="y")

    def plotear_posterior_modelos(posterior: np.ndarray) -> None:
        """
        Grafica el posterior sobre los modelos (Navaja de Occam Bayesiana).

        Args:
            posterior: Array de probabilidades posteriores, shape (D,).
                       Resultado de calcular_posterior_modelos().
        """
        colores = plt.cm.tab10(np.linspace(0, 1, D))

        plt.figure(figsize=(10, 6))

        for grado in range(D):
            plt.bar(grado, posterior[grado], color=colores[grado], alpha=0.8)

        plt.xlabel("Grado del polinomio")
        plt.ylabel("P(Modelo | Datos)")
        plt.title("Posterior sobre modelos (Navaja de Occam Bayesiana)")
        plt.xticks(range(D))
        plt.grid(True, alpha=0.3, axis="y")

    def plotear_ajustes_MAP(
        modelos_map: List[dict],
        X: np.ndarray,
        Y: np.ndarray,
        X_grilla: np.ndarray,
        Y_grilla: np.ndarray,
    ) -> None:
        """
        Grafica todas las curvas ajustadas por MAP superpuestas.

        Args:
            modelos_map: Lista de diccionarios con modelos MAP.
                         Resultado de ajustar_modelos_MAP().
            X: Datos de entrada, shape (N, 1)
            Y: Datos de salida, shape (N, 1)
            X_grilla: Grilla para evaluación, shape (100, 1)
            Y_grilla: Función verdadera en grilla, shape (100, 1)
        """
        plt.figure(figsize=(12, 8))

        colores = plt.cm.tab10(np.linspace(0, 1, D))

        plt.plot(
            X_grilla,
            Y_grilla,
            "--",
            color="black",
            label="Función verdadera",
            linewidth=2,
        )
        plt.scatter(X, Y, color="black", s=50, label="Datos", zorder=5)

        for modelo_info in modelos_map:
            coeficientes = modelo_info["media"]
            grado = modelo_info["grado"]
            predicciones = np.array(
                [
                    sum(coeficientes[i] * (x**i) for i in range(len(coeficientes)))
                    for x in X_grilla.ravel()
                ]
            )
            plt.plot(
                X_grilla,
                predicciones,
                color=colores[grado],
                label=f"Grado {grado}",
                alpha=0.8,
                linewidth=1.5,
            )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Media del posterior con prior no informativo")
        plt.ylim(-1.5, 1.5)
        plt.legend(ncol=3, fontsize=9)
        plt.grid(True, alpha=0.3)

    def plotear_prediccion_en_punto(
        X: np.ndarray, Y: np.ndarray, x_nuevo: float = -0.23
    ) -> None:
        """
        Grafica la distribución predictiva en un punto específico.

        Compara cómo tres modelos de diferente complejidad predicen
        en el mismo punto x = -0.23, usando solo los primeros 4 datos.

        Args:
            X: Datos de entrada, shape (N, 1)
            Y: Datos de salida, shape (N, 1)
            x_nuevo: Punto donde hacer la predicción (default -0.23)
        """
        grados_a_comparar = [0, 3, 9]
        y_rango = np.linspace(-2.5, 0.5, 300)

        plt.figure(figsize=(10, 6))

        colores = ["blue", "red", "cyan"]
        etiquetas = ["Rígido (grado 0)", "Simple (grado 3)", "Complejo (grado 9)"]

        # Usar solo primeros 4 datos (menos datos = más incertidumbre)
        datos_entrenamiento_X = X[0:4]
        datos_entrenamiento_Y = Y[0:4]

        for grado, color, etiqueta in zip(grados_a_comparar, colores, etiquetas):
            x_nuevo_array = np.array([[x_nuevo]])

            Phi_nuevo = phi(x_nuevo_array, grado).values
            Phi_entrenamiento = phi(datos_entrenamiento_X, grado).values

            densidades = []
            for y_val in y_rango:
                prediccion = ml.predictive(
                    np.array([y_val]),
                    Phi_nuevo,
                    alpha=ALPHA,
                    beta=BETA,
                    t_priori=datos_entrenamiento_Y.ravel(),
                    Phi_priori=Phi_entrenamiento,
                )
                densidades.append(prediccion)

            densidades = np.array(densidades)

            plt.plot(y_rango, densidades, color=color, linewidth=2, label=etiqueta)

        plt.xlabel(f"y | x = {x_nuevo}")
        plt.ylabel("P(y | x, Datos, Modelo)")
        plt.title("Distribución predictiva: modelo rígido vs simple vs complejo")
        plt.xlim(-2.5, 0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # =========================================================================
    # PASO 1: GENERACIÓN DE DATOS
    # =========================================================================
    print("=" * 70)
    print("PRÁCTICA 2.1: MÉTODOS DE INFERENCIA EXACTA")
    print("=" * 70)

    print("\n1. Generando datos...")
    X, Y, X_grilla, Y_grilla = generar_datos()
    print(f"   - N = {N} datos de entrenamiento")
    print(f"   - Dimensión de datos: X {X.shape}, Y {Y.shape}")

    print("\n2. Visualizando función objetivo y datos...")
    plotear_datos(X, Y, X_grilla, Y_grilla)
    plt.show()

    # =========================================================================
    # PASO 2: MÁXIMA VEROSIMILITUD (OLS)
    # =========================================================================
    print("\n3. Ajustando modelos por Máxima Verosimilitud (OLS)...")
    modelos_OLS = ajustar_modelos_OLS(X, Y)
    print(f"   - Modelos ajustados: grados 0 a {D - 1}")

    print("\n4. Analizando verosimilitudes...")
    log_likelihoods = extraer_verosimilitudes_OLS(modelos_OLS)
    plotear_verosimilitudes(log_likelihoods)
    plt.show()

    print("\n5. Visualizando ajustes OLS...")
    plotear_ajustes_OLS(modelos_OLS, X, Y, X_grilla, Y_grilla)
    plt.show()

    plotear_mejor_ajuste_OLS(modelos_OLS, X, Y, X_grilla, Y_grilla)
    plt.show()

    # =========================================================================
    # PASO 3: PREDICCIÓN ONLINE (OLS)
    # =========================================================================
    print("\n6. Evaluación de predicción online (OLS)...")
    log_evidencias_online = evaluacion_online_OLS(X, Y)
    plotear_evaluacion_online_OLS(log_evidencias_online)
    plt.show()

    # =========================================================================
    # PASO 4: EVIDENCIA BAYESIANA Y SELECCIÓN DE MODELOS
    # =========================================================================
    print("\n7. Calculando evidencia Bayesiana (Log-evidence)...")
    posterior = calcular_posterior_modelos(X, Y)
    plotear_posterior_modelos(posterior)
    plt.show()

    print("\n   Posterior sobre modelos (Navaja de Occam Bayesiana):")
    for grado in range(D):
        print(f"   - Grado {grado}: {posterior[grado]:.6f}")

    # =========================================================================
    # PASO 5: DISTRIBUCIÓN PREDICTIVA
    # =========================================================================
    print("\n8. Ajustando modelos por Máximo A Posteriori (MAP)...")
    modelos_MAP = ajustar_modelos_MAP(X, Y, alpha_factor=100)
    print(f"   - Prior regularizador aplicado (alpha_factor = 100)")

    #plotear_ajustes_MAP(modelos_MAP, X, Y, X_grilla, Y_grilla)
    #plt.show()

    print("\n9. Analizando distribución predictiva...")
    plotear_prediccion_en_punto(X, Y, x_nuevo=-0.23)
    plt.show()

    # =========================================================================
    # PASO 6: EJERCICIO DE ALTURAS (EFECTO CAUSAL DEL SEXO BIOLÓGICO)
    # =========================================================================
    print(
        "\n10. Ejercicio de alturas: efecto causal del sexo biológico sobre la altura"
    )
    try:
        df_alturas = pd.read_csv("CSVs/alturas.csv")
        print(f"   - Datos de alturas cargados: {len(df_alturas)} filas")
    except Exception as e:
        print(f"   - Error al cargar alturas.csv: {e}")
        df_alturas = None

    if df_alturas is not None:
        print("\n   Comparando modelos causales alternativos...")
        resultados_altura = comparar_modelos_altura(df_alturas)
        print("   - Evidencia modelo base:", resultados_altura["evidencia_base"])
        print(
            "   - Evidencia modelo biológico:", resultados_altura["evidencia_biologico"]
        )
        print(
            "   - Evidencia modelo identitario:",
            resultados_altura["evidencia_identitario"],
        )
        print("   - Mejor modelo:", resultados_altura["mejor_modelo"])
        print("   - Posterior modelos:", resultados_altura["posterior_modelos"])

    print("\n" + "=" * 70)
    print("Práctica completada exitosamente")
    print("=" * 70)
