#Para este modelo contamos con 3 tests y 4 comunidades.
#Se determinan 8 resultados posibles, segun los test para cada paciente:
# 000, 001, 010, 011, 100, 101, 110, 111

#El modelo busca estimar:

#pc: Prevalencia de cada comunidad
#st: Sensibilidad de cada test
#xt: Especificidad de cada test

#Para este modelo existe un supuesto fundamental de independencia condicional entre los test.

from multiprocessing import freeze_support

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def generar_datos_modelo_1(N, prevalencias):

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
    #Generar los datos
    datos, verdaderos = generar_datos_modelo_1(
        N=3000,
        prevalencias=[0.2, 0.4, 0.6, 0.8]
    )

    print("Datos observados:")

    for c in range(len(datos["r"])):

        print("\nComunidad", c + 1)

        for i in range(8):
            print(
                datos["etiquetas"][i],
                ":", datos["r"][c][i]
            )

    print("\nValores verdaderos:")
    print("Prevalencias:", verdaderos["prevalencias"])
    print("Sensibilidades:", verdaderos["sensibilidad"])
    print("Especificidades:", verdaderos["especificidad"])

    #Cantidad de comunidades
    n_comunidades = len(verdaderos["prevalencias"])
    N = np.full(n_comunidades, datos["N"])

    #Modelo 1
    # Hui-Walter presupone independencia condicional. Se calcula las probabilidades conjuntas como productos de probabilidades individuales.
    with pm.Model() as modelo_1:

        #Durante el modelado tuve problemas de convergencia en el modelo:   R_hat de los parametros = 1.53 (muy alto) y pocas muestras efectivas...
        #Entiendo que se debe al intercambio de etiquetas por lo que modifico la forma en la que se expresa uno de los parametros:

        #Prevalencia de cada comunidad
        p = pm.Beta("p", alpha=1, beta=1, shape=n_comunidades)

        #Especificidades de los tres tests
        x = pm.Beta("x", alpha=1, beta=1, shape=3)

        #Sensibilidades de los tres tests
        delta = pm.Beta("delta", alpha=1, beta=1, shape=3)
        s = pm.Deterministic("s", (1 - x) + x * delta)

        #Aunque de esta forma estamos suponiendo que los 3 tests funcionan mejor que el azar.

        #Probabilidades de los patrones para pacientes enfermos
        q_enfermos = pm.math.stack([
            (1-s[0])*(1-s[1])*(1-s[2]),  # 000
            s[0]*(1-s[1])*(1-s[2]),      # 100
            (1-s[0])*s[1]*(1-s[2]),      # 010
            s[0]*s[1]*(1-s[2]),          # 110
            (1-s[0])*(1-s[1])*s[2],      # 001
            s[0]*(1-s[1])*s[2],          # 101
            (1-s[0])*s[1]*s[2],          # 011
            s[0]*s[1]*s[2]               # 111
        ])

        #Probabilidades de los patrones para pacientes sanos
        q_sanos = pm.math.stack([
            x[0]*x[1]*x[2],                  # 000
            (1-x[0])*x[1]*x[2],              # 100
            x[0]*(1-x[1])*x[2],              # 010
            (1-x[0])*(1-x[1])*x[2],          # 110
            x[0]*x[1]*(1-x[2]),              # 001
            (1-x[0])*x[1]*(1-x[2]),          # 101
            x[0]*(1-x[1])*(1-x[2]),          # 011
            (1-x[0])*(1-x[1])*(1-x[2])      # 111
        ])

        #Sumo ambas posibilidades según la prevalencia y estimo la probabilidad de q
        q = (p[:, None] * q_enfermos[None, :] + (1-p[:, None]) * q_sanos[None, :])
        pm.Deterministic("q", q)

        #Distribución de los conteos observados
        pm.Multinomial(
            "diagnosticos",
            n=N,
            p=q,
            observed=datos["r"]
        )

        trace_1 = pm.sample(2000, tune=2000, chains=4, random_seed=111, target_accept=0.95)

    # Resumen de las estimaciones
    resumen = az.summary(
        trace_1,
        var_names=["p", "s", "x"],
        ci_prob=0.89
    )

    print("\nResumen de las estimaciones:")
    print(resumen)

    # Control de divergencias
    divergencias = trace_1.sample_stats["diverging"].values.sum()

    print("\nCantidad de divergencias:", divergencias)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variables = ["p", "s", "x"]

    valores_reales = [
        verdaderos["prevalencias"],
        verdaderos["sensibilidad"],
        verdaderos["especificidad"]
    ]

    titulos = ["Prevalencias", "Sensibilidades", "Especificidades"]

    for i in range(3):
        variable = variables[i]
        nombres = [f"{variable}[{j}]" for j in range(len(valores_reales[i]))]

        datos_plot = resumen.loc[nombres]

        medias = datos_plot["mean"].values
        inferiores = datos_plot["eti89_lb"].values
        superiores = datos_plot["eti89_ub"].values

        posiciones = np.arange(len(nombres))

        axes[i].errorbar(
            posiciones,
            medias,
            yerr=[medias - inferiores, superiores - medias],
            fmt="o",
            capsize=4,
            label="Estimación e intervalo 89%"
        )

        axes[i].scatter(
            posiciones,
            valores_reales[i],
            marker="x",
            s=80,
            label="Valor verdadero"
        )

        if variable == "p":
            etiquetas = [f"C{j+1}" for j in range(len(nombres))]
        else:
            etiquetas = [f"Test {j+1}" for j in range(len(nombres))]

        axes[i].set_xticks(posiciones)
        axes[i].set_xticklabels(etiquetas)
        axes[i].set_ylim(0, 1)
        axes[i].set_title(titulos[i])
        axes[i].set_ylabel("Valor")
        axes[i].grid(alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.show()

    #Converge correctamente
    az.plot_trace(trace_1, var_names=["p", "s", "x"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()

#Conclusiones:
#El modelo converge correctamente, dado que los parametros tienen un r_hat = 1 y muestras efectivas suficientes...
#Se puede observar en los graficos como el modelo genera 4 estimaciones que subestiman los valores verdaderos de las prevalencias.
#Para los test 1 y 2, el modelo sobrestima considerablemente sus estimaciones de Sensibilidad.
#No parece haber problemas en la estimación de Especificidades de los test.