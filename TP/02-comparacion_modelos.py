import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# Datos: cantidad de positivos (0 a 3) de cada operador (fila) en cada contexto (columna)
data = pd.read_csv("TP/CSVs/02-positivos_por_cepas_verdaderos.csv")
contextos = ["20par-DTU1", "20par-DTU2", "20par-DTU6", "50par-DTU1", "50par-DTU2", "50par-DTU6"]
N = 3
n = np.round(data[contextos].to_numpy() / 100 * N).astype(int)

# Modelos
# Todos comparten el desempeño de cada contexto, p_c ~ Beta(2,2).
# Cada función crea los parámetros propios del modelo y devuelve log P(Datos | parámetros).

def base(p):
    # n_ci ~ Binomial(3, p_c)
    return pm.logp(pm.Binomial.dist(n=N, p=p), n).sum()

def operadores(p):
    # q ~ Beta(1,9): probabilidad de que un operador trabaje mal
    # r ~ Beta(1,1): factor que reduce el desempeño cuando el operador trabaja mal
    # C_i ~ Bernoulli(q), sumado sobre sus dos valores:
    #   P(n_i) = (1-q) prod_c Binomial(n_ci | 3, p_c) + q prod_c Binomial(n_ci | 3, p_c r)
    q = pm.Beta("q", alpha=1, beta=9)
    r = pm.Beta("r", alpha=1, beta=1)
    log_bien = pt.log(1 - q) + pm.logp(pm.Binomial.dist(n=N, p=p), n).sum(axis=1)
    log_mal = pt.log(q) + pm.logp(pm.Binomial.dist(n=N, p=p * r), n).sum(axis=1)
    return pt.logaddexp(log_bien, log_mal).sum()

#Modelo Alternativo Propuesto
#Estructura y Decisiones tomadas:

# 1. Relaciono el desempeño de baja y alta concentración por cada cepa, mediante los parámetros p_low, rho y delta.
# 2. Aplico partial pooling entre las cepas de alta concentración, mediante delta y sigma_H, permitiendo que
# cada cepa conserve un comportamiento particular mediante epsilon.
# 3. Modelo diferencias entre operadores mediante un r_i específico para cada uno, compartiendo información
# entre ellos mediante mu_r y sigma_r, con una probabilidad de error general q_error

#El modelo se divide en 3 pares principales.

def modelo_alternativo(p):
    #PARTE 1 | modelado de cepas:
    #Una distribucion Beta (poco informativa) como prior para el desempeño de las 3 cepas en concentración baja:
    p_low = pm.Beta(
        "p_low",
        alpha=2,
        beta=2,
        shape=3
    )

    #Un prior Gamma para modelar un efecto positivo del aumento de la concentración sobre el desempeño, en escala logit:
    delta = pm.Gamma(
        "delta",
        alpha=2,
        beta=1
    )

    #Un prior para rho Beta (no informativo) para aprender cuánto de las diferencias observadas en baja concentración se conserva en alta:
    rho = pm.Beta(
        "rho",
        alpha=1,
        beta=1
    )

    #Una misma distribucion con variabilidad para modelar en efecto en alta concentración, con un prior Normal:
    sigma_H = pm.HalfNormal(
        "sigma_H",
        sigma=1
    )

    epsilon_raw = pm.Normal(
        "epsilon_raw",
        mu=0,
        sigma=1,
        shape=3
    )
    
    epsilon = pm.Deterministic(
        "epsilon", sigma_H * epsilon_raw
    )

    #Propongo: "logit(pH,j​) = ρlogit(pL,j​) + Δ + ϵj​" para modelar todas las relaciones entre parámetros.
    logit_low = pt.log(p_low) - pt.log1p(-p_low)

    logit_high = (rho * logit_low + delta + epsilon)

    p_high = pm.Deterministic(
        "p_high", pm.math.sigmoid(logit_high)
    )

    #Un vector para los 6 contextos
    p_contextos = pt.concatenate([p_low, p_high])

    #PARTE 2 | Modelado de Operadores:
    #Se representa la probabilidad de error de los trabajadores como q, con un prior con distribucion Beta(1, 9) (lo mantengo igual que en el modelo "operadores")
    q_error = pm.Beta(
        "q_error",
        alpha=1,
        beta=9
    )

    #Modelado jerarquico para r, definiendo una gravedad de error distinta para cada operador:
    mu_r = pm.Normal(
        "mu_r",
        mu=0,
        sigma=2
    )

    sigma_r = pm.HalfNormal(
        "sigma_r",
        sigma=1
    )

    z_r_raw = pm.Normal("z_r_raw", mu=0, sigma=1, shape=14)
    z_r = mu_r + sigma_r * z_r_raw

    #Transformamos la escala de [-inf, inf] a [0, 1]
    r_i = pm.Deterministic("r_i", pm.math.sigmoid(z_r))

    #Función por partes para modelar el error del operador:
    log_bien = (pt.log(1 - q_error) + pm.logp(pm.Binomial.dist(n=N, p=p_contextos), n).sum(axis=1))

    log_mal = (pt.log(q_error) + pm.logp(pm.Binomial.dist(n=N, p=p_contextos * r_i[:, None]), n).sum(axis=1))

    #Se calcula la probabilidad de observar todos los resultados
    #Devolvemos la log-verosimilitud obteniendola de una Binomial; dada por cada operador, contexto y probabilidad de error.
    return pt.logaddexp(log_bien, log_mal).sum()

modelos = {"Base": base, "Operadores": operadores, "Alternativo": modelo_alternativo}  # Para agregar un modelo, sumarlo acá

# Selector de modelos
# M es una variable que elige qué modelo genera los datos. Por la regla de Bayes,
#   P(M | Datos) ∝ P(Datos | M) P(M)
# así que la evidencia de cada modelo es, salvo una constante, P(M | Datos) / P(M).

def selector(prior_M, muestras=10000):
    with pm.Model():
        p = pm.Beta("p", alpha=2, beta=2, shape=len(contextos))
        log_verosimilitudes = pt.stack([modelo(p) for modelo in modelos.values()])
        M = pm.Categorical("M", p=prior_M)
        pm.Potential("verosimilitud", log_verosimilitudes[M])
        traza = pm.sample(muestras, tune=2000, chains=4, target_accept=0.99, random_seed=42)
    M = traza.posterior["M"].values.ravel()
    return np.array([np.mean(M == k) for k in range(len(modelos))])

# Si un modelo es mucho mejor que otro, el selector casi nunca visita al peor y la
# estimación es mala. Por eso primero lo corremos con un prior uniforme y después
# le damos más prior a los modelos poco visitados, hasta que los visite a todos.

if __name__ == "__main__":
    prior_M = np.ones(len(modelos)) / len(modelos)
    for intento in range(3):
        posterior_M = selector(prior_M)
        log_evidencia = np.log(np.maximum(posterior_M, 1e-4)) - np.log(prior_M)  # salvo una constante
        print(f"\nPrior de M: {np.round(prior_M, 4)}   Posterior de M: {np.round(posterior_M, 4)}")
        if posterior_M.min() > 0.05:
            break
        prior_M = np.exp(-log_evidencia) / np.exp(-log_evidencia).sum()


    # Resultados
    log_evidencia -= log_evidencia.max()
    print("\nlog evidencia relativa al mejor modelo:")
    for nombre, le in zip(modelos, log_evidencia):
        print(f"  {nombre:12s} {le:8.2f}")
    print("\nP(Modelo | Datos) con prior uniforme entre modelos:")
    for nombre, pm_ in zip(modelos, np.exp(log_evidencia) / np.exp(log_evidencia).sum()):
        print(f"  {nombre:12s} {pm_:.4f}")