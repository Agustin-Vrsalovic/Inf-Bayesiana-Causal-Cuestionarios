# Parámetros verdaderos:
# p | prevalencia de la enfermedad
# s | sensibilidad del test
# x | especificidad del test

# Probabilidad de test positivo: q_1 = p*s + (1-p)*(1-x)
# Probabilidad de test negativo: q_0 = 1 - q_1

from multiprocessing import freeze_support

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def generar_datos_modelo_0(N, p, s, x):
    # Excepciones
    if N <= 0:
        raise ValueError("N debe ser positivo.")

    if not all(0 <= valor <= 1 for valor in [p, s, x]):
        raise ValueError("p, s y x deben estar entre 0 y 1.")

    rng = np.random.default_rng(111)

    #Simular el estado real de cada individuo
    estado_real = rng.binomial(1, p, size=N)

    #Probabilidad de resultado positivo para cada individuo
    prob_positivo = np.zeros(N)

    for i in range(N):
        if estado_real[i] == 1:
            prob_positivo[i] = s
        else:
            prob_positivo[i] = 1 - x

    #Simular los resultados del test
    diagnostico = rng.binomial(1, prob_positivo)

    #Conteo de resultados
    n0 = np.sum(diagnostico == 0)
    n1 = np.sum(diagnostico == 1)

    #Datos observables
    datos = {"N": N, "r": np.array([n0, n1]),
             "n0": n0, "n1": n1}

    #Valores verdaderos
    q1 = p * s + (1 - p) * (1 - x)

    verdaderos = {
        "p": p,
        "s": s,
        "x": x,
        "q1": q1
    }

    return datos, verdaderos


def main():

    #Genero los datos
    datos, verdaderos = generar_datos_modelo_0(
        N=1000, p=0.4, s=0.9, x=0.85
    )

    print("Datos observados:")
    print(datos)

    print("\nValores verdaderos:")
    print(verdaderos)

    # Modelo 0
    with pm.Model() as modelo_0:

        p = pm.Beta("p", alpha=1, beta=1)
        s = pm.Beta("s", alpha=1, beta=1)
        x = pm.Beta("x", alpha=1, beta=1)

        q1 = pm.Deterministic("q1", p * s + (1 - p) * (1 - x))

        q = pm.math.stack([1 - q1, q1])

        pm.Multinomial(
            "diagnosticos",
            n=datos["N"],
            p=q,
            observed=datos["r"]
        )

        trace_0 = pm.sample(2000, tune=1000, chains=4, random_seed=111, target_accept=0.9)

    # Resumen de las estimaciones
    resumen = az.summary(trace_0, var_names=["p", "s", "x", "q1"])
    print("\nResumen de las estimaciones:")
    print(resumen)

    # Generar predicciones posteriores

    predicciones = pm.sample_posterior_predictive(
        trace_0,
        model=modelo_0,
        var_names=["diagnosticos"],
        random_seed=111
    )

    conteos_predichos = predicciones.posterior_predictive[
        "diagnosticos"
    ].values

    conteos_predichos = conteos_predichos.reshape(-1, 2)

    # Calcular mediana e intervalo central del 89%
    mediana = np.median(conteos_predichos, axis=0)
    limite_inferior = np.quantile(conteos_predichos, 0.055, axis=0)
    limite_superior = np.quantile(conteos_predichos, 0.945, axis=0)

    # Graficar resultados
    posiciones = np.array([0, 1])

    plt.figure(figsize=(7, 5))

    plt.bar(
        posiciones,
        datos["r"],
        label="Datos observados",
        alpha=0.6
    )

    plt.errorbar(
        posiciones,
        mediana,
        yerr=[
            mediana - limite_inferior,
            limite_superior - mediana
        ],
        fmt="o",
        color="red",
        capsize=5,
        label="Predicción posterior (89%)"
    )

    plt.xticks(posiciones, ["Negativos", "Positivos"])
    plt.ylabel("Cantidad de pacientes")
    plt.title("Predicción de resultados - Modelo 0")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()

#Conclusiones
#No parecen haber problemas de Muestreo, el modelo converge correctamente
#Como era de imaginar... el modlo estima correctamente 'q1' (probabilidad de que el test dé positivo),
# mientras que p, s y x tienen intervalos muy amplio; sin poder identificar sus efectos por separado.