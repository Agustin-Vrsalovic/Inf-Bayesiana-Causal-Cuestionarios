#Harmony plantea entender todas las relaciones entre test (y no unicamente de 2 en 2), incorporando las covarianzas de los 3 pares posibles (para 3 test)
#Para cada comunidad, la probabilidad de observar un patrón combina 2 posibilidades:
# 1. Que el individuo esté enfermo, ponderada por la prevalencia.
# 2. Que esté sano, ponderada por 1 - (la prevalencia).

#El problema de Harmony es que contabiliza dos veces la covarianza, lo que representa un problema para las estimaciones del modelo.

from multiprocessing import freeze_support

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

import pytensor.tensor as pt

#Mismo simulador que el modelo 2
def generar_datos_modelo_3(N, prevalencias):
    #Excepciones
    if N <= 0:
        raise ValueError("N debe ser positivo.")

    if len(prevalencias) < 2:
        raise ValueError("Se necesitan al menos dos comunidades.")

    if not all(0 <= p <= 1 for p in prevalencias):
        raise ValueError("Las prevalencias deben estar entre 0 y 1.")

    rng = np.random.default_rng(111)

    #Valores verdaderos de los tests
    sensibilidad = np.array([0.80, 0.75, 0.70])

    #Especificidades elegidas
    especificidad = np.array([0.90, 0.85, 0.80])

    #Probabilid de falso positivo
    falso_positivo = 1 - especificidad

    #Distribución conjunta de los resultados de los tests para pacientes enfermos
    
    #El índice representa el patrón:
    # bit 0: test 1
    # bit 1: test 2
    # bit 2: test 3

    prob_enfermos = np.array([
        0.045,  # 000
        0.030,  # 100
        0.015,  # 010
        0.210,  # 110
        0.125,  # 001
        0.050,  # 101
        0.015,  # 011
        0.510   # 111
    ])

    #Etiquetas
    etiquetas = [
        "000", "100", "010", "110",
        "001", "101", "011", "111"
    ]

    #Conteos por cada comunidad
    conteos = np.zeros((len(prevalencias), 8), dtype=int)

    #Simular cada comunidad
    for c in range(len(prevalencias)):

        #Estado real de cada paciente
        estado_real = rng.binomial(1, prevalencias[c], size=N)

        #Patrón de diagnóstico de cada paciente
        patrones = np.zeros(N, dtype=int)

        for i in range(N):
            if estado_real[i] == 1: # Paciente enfermo
                patron = rng.choice(8, p=prob_enfermos)
            else: # Paciente sano
                diagnosticos = rng.binomial(1, falso_positivo, size=3)

                patron = (
                    diagnosticos[0]
                    + 2 * diagnosticos[1]
                    + 4 * diagnosticos[2]
                )

            patrones[i] = patron

        #Conteo de cada patrón
        conteos[c] = np.bincount(patrones, minlength=8)

    #Datos observables
    datos = {
        "N": N,
        "r": conteos,
        "etiquetas": etiquetas
    }

    #Valores verdaderos
    cov_s = np.array([
        [0.00, 0.12, 0.00],
        [0.12, 0.00, 0.00],
        [0.00, 0.00, 0.00]
    ])

    cov_x = np.zeros((3, 3))

    verdaderos = {
        "prevalencias": np.array(prevalencias),
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "cov_s": cov_s,
        "cov_x": cov_x
    }

    return datos, verdaderos

def main():

    datos, valores_reales = generar_datos_modelo_3(N=3000, prevalencias=[0.2, 0.4, 0.6, 0.8])

    r = datos["r"]
    N = datos["N"]
    etiquetas = datos["etiquetas"]

    print("Dimensiones de los conteos:", r.shape)
    print("Totales por comunidad:", r.sum(axis=1))
    print("Etiquetas:", datos["etiquetas"])

    patrones = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])

    pares = [(0, 1), (0, 2), (1, 2)]

    def calcular_probabilidades_harmony(marginales, covarianzas):
        probabilidades = []

        for patron in patrones:

            prob = 1.0

            for t in range(3):
                if patron[t] == 1:
                    prob = prob * marginales[t]
                else:
                    prob = prob * (1 - marginales[t])

            for k, (i, j) in enumerate(pares):
                if patron[i] == patron[j]:
                    prob = prob + covarianzas[k]
                else:
                    prob = prob - covarianzas[k]

            probabilidades.append(prob)

        return pt.stack(probabilidades)

    def calcular_factor_maximo_harmony(marginales, covarianzas):

        prob_base = calcular_probabilidades_harmony(
            marginales,
            pt.zeros_like(covarianzas)
        )

        prob_sin_restriccion = calcular_probabilidades_harmony(
            marginales,
            covarianzas
        )

        ajuste = prob_sin_restriccion - prob_base

        limites = pt.switch(
            ajuste < 0,
            prob_base / pt.maximum(-ajuste, 1e-12),
            pt.ones_like(ajuste)
        )

        return pt.minimum(1.0, pt.min(limites))

    #Las reglas que sugiere el modelo son:
    # Partir del producto de las probabilidades marginales
    # Sumar la covarianza de cada par cuando los 2 resultados coinciden y restar la covarianza cuando los resultados son distintos.
    
    #Modelo 3
    with pm.Model() as modelo_3:

        #Prevalencias por comunidad
        p = pm.Beta("p", alpha=1, beta=1, shape=4)

        #Especificidades
        x = pm.Beta("x", alpha=1, beta=1, shape=3)

        #Sensibilidades
        delta = pm.Beta("delta", alpha=1, beta=1, shape=3)
        s = pm.Deterministic("s", (1 - x) + x * delta)

        #Covarianzas entre sensibilidades
        covse_min = []
        covse_max = []

        #Covarianzas entre especificidades
        covsp_min = []
        covsp_max = []

        for i, j in pares:

            minimo_s = pt.maximum(-s[i] * s[j], -(1 - s[i]) * (1 - s[j]))
            maximo_s = pt.minimum(s[i], s[j]) - s[i] * s[j]

            covse_min.append(minimo_s)
            covse_max.append(maximo_s)

            minimo_x = pt.maximum(-x[i] * x[j], -(1 - x[i]) * (1 - x[j]))
            maximo_x = pt.minimum(x[i], x[j]) - x[i] * x[j]

            covsp_min.append(minimo_x)
            covsp_max.append(maximo_x)

        #Covarianzas iniciales dentro de sus intervalos admisibles
        u_covse = pm.Beta("u_covse", alpha=1, beta=1, shape=3)
        u_covsp = pm.Beta("u_covsp", alpha=1, beta=1, shape=3)

        covse_base = (
            pt.stack(covse_min) +
            (pt.stack(covse_max) - pt.stack(covse_min)) * u_covse
        )

        covsp_base = (
            pt.stack(covsp_min) +
            (pt.stack(covsp_max) - pt.stack(covsp_min)) * u_covsp
        )

        #Factores para mantener positivas las probabilidades conjuntas
        rho_covse = pm.Beta("rho_covse", alpha=1, beta=1)
        rho_covsp = pm.Beta("rho_covsp", alpha=1, beta=1)

        factor_se = calcular_factor_maximo_harmony(s, covse_base)
        factor_sp = calcular_factor_maximo_harmony(1 - x, covsp_base)

        covse = pm.Deterministic("covse", covse_base * factor_se * rho_covse)

        covsp = pm.Deterministic("covsp", covsp_base * factor_sp * rho_covsp)

        #Probabilidades de los patrones
        prob_enfermos = calcular_probabilidades_harmony(s, covse)
        prob_sanos = calcular_probabilidades_harmony(1 - x, covsp)

        #Verosimilitud por comunidad
        for c in range(4):

            q = p[c] * prob_enfermos + (1 - p[c]) * prob_sanos

            pm.Multinomial(
                f"observaciones_{c+1}",
                n=int(r[c].sum()),
                p=q,
                observed=r[c]
            )

        #Inferencia
        trace_harmony = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            init="adapt_diag",
            random_seed=111
        )

    resumen_harmony = az.summary(
        trace_harmony,
        var_names=["p", "s", "x", "covse", "covsp"],
        ci_prob=0.89
    )

    print(resumen_harmony)

    print("Cantidad de divergencias:", int(trace_harmony.sample_stats["diverging"].sum()))

    print("R-hat máximo:", resumen_harmony["r_hat"].max())


    #Graficos de Harmony: valores verdaderos contra posteriores
    def graficar_parametro(ax, nombre, valores_reales, etiquetas, titulo):

        indices = [f"{nombre}[{i}]" for i in range(len(valores_reales))]

        medias = resumen_harmony.loc[indices, "mean"].to_numpy()
        inferiores = resumen_harmony.loc[indices, "eti89_lb"].to_numpy()
        superiores = resumen_harmony.loc[indices, "eti89_ub"].to_numpy()

        errores = np.array([
            medias - inferiores,
            superiores - medias
        ])

        posiciones = np.arange(len(indices))

        ax.errorbar(
            medias,
            posiciones,
            xerr=errores,
            fmt="o",
            capsize=4,
            label="Posterior (media e intervalo 89%)"
        )

        ax.scatter(
            valores_reales,
            posiciones,
            marker="|",
            s=180,
            label="Valor verdadero"
        )

        ax.set_yticks(posiciones)
        ax.set_yticklabels(etiquetas)
        ax.invert_yaxis()
        ax.set_title(titulo)
        ax.set_xlabel("Valor del parámetro")
        ax.grid(axis="x", alpha=0.3)


    #Prevalencias, sensibilidades y especificidades
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    graficar_parametro(
        axes[0],
        "p",
        [0.2, 0.4, 0.6, 0.8],
        ["Comunidad 1", "Comunidad 2", "Comunidad 3", "Comunidad 4"],
        "Prevalencias"
    )

    graficar_parametro(
        axes[1],
        "s",
        [0.8, 0.75, 0.7],
        ["Test 1", "Test 2", "Test 3"],
        "Sensibilidades"
    )

    graficar_parametro(
        axes[2],
        "x",
        [0.9, 0.85, 0.8],
        ["Test 1", "Test 2", "Test 3"],
        "Especificidades"
    )

    axes[0].legend(fontsize=8)
    fig.suptitle("Harmony: valores verdaderos y estimaciones posteriores")
    plt.tight_layout()
    plt.savefig("modelo_3_parametros.png", dpi=300, bbox_inches="tight")
    plt.show()


    #Covarianzas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    etiquetas_pares = ["Tests 1-2", "Tests 1-3", "Tests 2-3"]

    graficar_parametro(
        axes[0],
        "covse",
        [0.12, 0.0, 0.0],
        etiquetas_pares,
        "Covarianzas en enfermos"
    )

    graficar_parametro(
        axes[1],
        "covsp",
        [0.0, 0.0, 0.0],
        etiquetas_pares,
        "Covarianzas en sanos"
    )

    axes[0].legend(fontsize=8)
    fig.suptitle("Harmony: covarianzas verdaderas y posteriores")
    plt.tight_layout()
    plt.savefig("modelo_3_covarianzas.png", dpi=300, bbox_inches="tight")
    plt.show()


    #Comprobacion de si los intervalos contienen los valores verdaderos

    parametros = {
        "p": [0.2, 0.4, 0.6, 0.8],
        "s": [0.8, 0.75, 0.7],
        "x": [0.9, 0.85, 0.8],
        "covse": [0.12, 0.0, 0.0],
        "covsp": [0.0, 0.0, 0.0]
    }

    print("\nComparacion con los valores verdaderos:")

    for nombre, valores in parametros.items():

        for i, verdadero in enumerate(valores):

            fila = resumen_harmony.loc[f"{nombre}[{i}]"]

            incluido = (
                fila["eti89_lb"] <= verdadero <= fila["eti89_ub"]
            )

            print(
                f"{nombre}[{i}]: "
                f"Verdadero={verdadero:.3f}, "
                f"Posterior={fila['mean']:.3f}, "
                f"Intervalo=[{fila['eti89_lb']:.3f}, "
                f"{fila['eti89_ub']:.3f}], "
                f"Contiene verdadero={incluido}"
            )

if __name__ == "__main__":
    freeze_support()
    main()


#Se consiguió una buena convergencia ya que el modelo obtuvo 0 divergencias, un R-hat máximo de 1.0019 y tamaños efectivos de muestra superiores a 1300.
#Las prevalencias y especificidades fueron subestimadas respecto de sus valores verdaderos; mientras que las sensibilidades resultaron sobrestimadas.
#Esto significa que Harmony no logra recuperar correctamente los parámetros con los que se generaron los datos.