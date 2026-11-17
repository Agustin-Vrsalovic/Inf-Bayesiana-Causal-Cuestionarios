#A diferencia del anterior, en este modelo ya no suponemos que los tests son condicionalmente independientes.
#Dendukuri agrega dos parámetros que permiten representar la dependencia entre un par de tests:
# cs: covarianza entre los resultados positivos de ambos tests en individuos enfermos
# cx: covarianza entre los resultados negativos de ambos tests en individuos sanos

#Considero covarianzas comunes entre las cuatro comunidades y prevalencia distinta según la comunidad.
#Como teniamos antes podemos diferenciar los resultados por patrones: 00, 01, 10, 11

#La probabilidad de observar un patrón para este modelo es:
# qc = p_c * [(1 - s_a)(1 - sb) + cs]   +   (1 - pc)[xa*xb + cx]


from multiprocessing import freeze_support

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

#Mismo simulador que el modelo 1
def generar_datos_modelo_2(N, prevalencias):

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
    #Genero los datos
    datos, verdaderos = generar_datos_modelo_2(N=3000, prevalencias=[0.2, 0.4, 0.6, 0.8])

    r = datos["r"]

    #Separamos conteos para los tests 1 y 2
    conteos_12 = np.column_stack([
        r[:, 0] + r[:, 4], #00
        r[:, 2] + r[:, 6], #01
        r[:, 1] + r[:, 5], #10
        r[:, 3] + r[:, 7]  #11
    ])

    #Separamos conteos para los tests 1 y 3
    conteos_13 = np.column_stack([
        r[:, 0] + r[:, 2], #00
        r[:, 4] + r[:, 6], #01
        r[:, 1] + r[:, 3], #10
        r[:, 5] + r[:, 7]  #11
    ])

    print("Conteos para los tests 1-2:")
    print(conteos_12)

    print("\nConteos para los tests 1-3:")
    print(conteos_13)

    print("\nTotales por comunidad:")
    print(conteos_12.sum(axis=1))
    print(conteos_13.sum(axis=1))

    n_comunidades = len(verdaderos["prevalencias"])
    N = np.full(n_comunidades, datos["N"])

    #Dendukuri representa las covarianzas entre un par de test
    #Modelo 2 | Test 1 y 2

    with pm.Model() as modelo_2:

        #Prevalencias por comunidad
        p = pm.Beta("p", alpha=1, beta=1, shape=n_comunidades)

        #Especificidades de los tests
        x = pm.Beta("x", alpha=1, beta=1, shape=2)

        #Sensibilidades de los tests
        delta = pm.Beta("delta", alpha=1, beta=1, shape=2)
        s = pm.Deterministic("s", (1 - x) + x * delta)

        #Límites posibles de la covarianza en enfermos
        cs_min = (s[0] - 1) * (1 - s[1])
        cs_max = pm.math.minimum(s[0], s[1]) - s[0] * s[1]

        #Covarianza en enfermos
        u_s = pm.Beta("u_s", alpha=1, beta=1)
        cs = pm.Deterministic("cs", cs_min + (cs_max - cs_min) * u_s)

        #Límites posibles de la covarianza en sanos
        cx_min = (x[0] - 1) * (1 - x[1])
        cx_max = pm.math.minimum(x[0], x[1]) - x[0] * x[1]

        # Covarianza en sanos
        u_x = pm.Beta("u_x", alpha=1, beta=1)
        cx = pm.Deterministic("cx", cx_min + (cx_max - cx_min) * u_x)

        #Probabilidades de los 4 patrones
        q0 = (
            p * ((1 - s[0]) * (1 - s[1]) + cs)
            + (1 - p) * (x[0] * x[1] + cx)
        )

        q1 = (
            p * ((1 - s[0]) * s[1] - cs)
            + (1 - p) * (x[0] * (1 - x[1]) - cx)
        )

        q2 = (
            p * (s[0] * (1 - s[1]) - cs)
            + (1 - p) * ((1 - x[0]) * x[1] - cx)
        )

        q3 = (
            p * (s[0] * s[1] + cs)
            + (1 - p) * ((1 - x[0]) * (1 - x[1]) + cx)
        )

        #Distribución conjunta por comunidad
        q = pm.math.stack([q0, q1, q2, q3], axis=1)

        pm.Deterministic("q", q)

        #Verosimilitud
        pm.Multinomial("diagnosticos", n=N, p=q, observed=conteos_12)

        # Inferencia
        trace_12 = pm.sample(
            2000,
            tune=2000,
            chains=4,
            random_seed=111,
            target_accept=0.99,
        )

    resumen_12 = az.summary(
        trace_12,
        var_names=["p", "s", "x", "cs", "cx"],
        ci_prob=0.89
    )

    print("\nResumen Dendukuri - Tests 1 y 2:")
    print(resumen_12)

    divergencias = trace_12.sample_stats["diverging"].values.sum()

    print("\nCantidad de divergencias:", divergencias)

    #Valores verdaderos
    print("\nValores verdaderos:")
    print("Prevalencias:", verdaderos["prevalencias"])
    print("Sensibilidades:", verdaderos["sensibilidad"][[0, 1]])
    print("Especificidades:", verdaderos["especificidad"][[0, 1]])
    print("Covarianza en enfermos:", verdaderos["cov_s"][0, 1])
    print("Covarianza en sanos:", verdaderos["cov_x"][0, 1])


    #Modelo 2 | Test 1 y 3

    with pm.Model() as modelo_13:

        #Prevalencias por comunidad
        p = pm.Beta("p", alpha=1, beta=1, shape=n_comunidades)

        #Especificidades de los tests
        x = pm.Beta("x", alpha=1, beta=1, shape=2)

        #Sensibilidades
        delta = pm.Beta("delta", alpha=1, beta=1, shape=2)
        s = pm.Deterministic("s", (1 - x) + x * delta)

        #Límites de la covarianza en enfermos
        cs_min = (s[0] - 1) * (1 - s[1])
        cs_max = pm.math.minimum(s[0], s[1]) - s[0] * s[1]

        #Covarianza en enfermos
        u_s = pm.Beta("u_s", alpha=1, beta=1)

        cs = pm.Deterministic("cs", cs_min + (cs_max - cs_min) * u_s)

        # Límites de la covarianza en sanos
        cx_min = (x[0] - 1) * (1 - x[1])
        cx_max = pm.math.minimum(x[0], x[1]) - x[0] * x[1]

        # Covarianza en sanos
        u_x = pm.Beta("u_x", alpha=1, beta=1)

        cx = pm.Deterministic("cx", cx_min + (cx_max - cx_min) * u_x)

        # Probabilidades de los cuatro patrones
        q0 = (
            p * ((1 - s[0]) * (1 - s[1]) + cs)
            + (1 - p) * (x[0] * x[1] + cx)
        )

        q1 = (
            p * ((1 - s[0]) * s[1] - cs)
            + (1 - p) * (x[0] * (1 - x[1]) - cx)
        )

        q2 = (
            p * (s[0] * (1 - s[1]) - cs)
            + (1 - p) * ((1 - x[0]) * x[1] - cx)
        )

        q3 = (
            p * (s[0] * s[1] + cs)
            + (1 - p) * ((1 - x[0]) * (1 - x[1]) + cx)
        )

        #Distribución conjunta por comunidad
        q = pm.math.stack([q0, q1, q2, q3], axis=1)

        pm.Deterministic("q", q)

        # Verosimilitud: ahora usamos los conteos 1-3
        pm.Multinomial(
            "diagnosticos",
            n=N,
            p=q,
            observed=conteos_13
        )

        # Inferencia
        trace_13 = pm.sample(
            2000,
            tune=2000,
            chains=4,
            target_accept=0.99,
            random_seed=111
        )

    # Resumen de las estimaciones
    resumen_13 = az.summary(
        trace_13,
        var_names=["p", "s", "x", "cs", "cx"],
        ci_prob=0.89
    )

    print("\nResumen Dendukuri - Tests 1 y 3:")
    print(resumen_13)

    # Verificación de convergencia
    divergencias = trace_13.sample_stats["diverging"].values.sum()

    print("\nCantidad de divergencias:", divergencias)

    rhat = az.rhat(trace_13)

    rhat_max = max(
        float(variable.max())
        for variable in rhat.data_vars.values()
    )

    print(f"R-hat máximo: {rhat_max:.4f}")

    # Valores verdaderos para comparar
    print("\nValores verdaderos:")
    print("Prevalencias:", verdaderos["prevalencias"])
    print("Sensibilidades:", verdaderos["sensibilidad"][[0, 2]])
    print("Especificidades:", verdaderos["especificidad"][[0, 2]])
    print("Covarianza en enfermos:", verdaderos["cov_s"][0, 2])
    print("Covarianza en sanos:", verdaderos["cov_x"][0, 2])


    #Graficos
    def graficar(ax, resumen, nombres, valores_reales, etiquetas, titulo):

        tabla = resumen.loc[nombres]

        medias = tabla["mean"].values
        inferiores = tabla["eti89_lb"].values
        superiores = tabla["eti89_ub"].values

        posiciones = np.arange(len(nombres))

        ax.errorbar(
            posiciones,
            medias,
            yerr=[medias - inferiores, superiores - medias],
            fmt="o",
            capsize=4,
            label="Estimación posterior"
        )

        ax.scatter(
            posiciones,
            valores_reales,
            marker="x",
            s=80,
            label="Valor verdadero"
        )

        ax.set_xticks(posiciones)
        ax.set_xticklabels(etiquetas)
        ax.set_title(titulo)
        ax.grid(alpha=0.3)


    def graficar_resultados(resumen, indices_tests, nombre_par):

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        etiquetas_tests = [
            f"Test {i + 1}" for i in indices_tests
        ]

        # Prevalencias
        graficar(
            axes[0, 0], resumen,
            ["p[0]", "p[1]", "p[2]", "p[3]"],
            verdaderos["prevalencias"],
            ["C1", "C2", "C3", "C4"],
            "Prevalencias"
        )

        # Sensibilidades
        graficar(
            axes[0, 1], resumen,
            ["s[0]", "s[1]"],
            verdaderos["sensibilidad"][indices_tests],
            etiquetas_tests,
            "Sensibilidades"
        )

        # Especificidades
        graficar(
            axes[1, 0], resumen,
            ["x[0]", "x[1]"],
            verdaderos["especificidad"][indices_tests],
            etiquetas_tests,
            "Especificidades"
        )

        # Covarianzas
        i, j = indices_tests

        graficar(
            axes[1, 1], resumen,
            ["cs", "cx"],
            [
                verdaderos["cov_s"][i, j],
                verdaderos["cov_x"][i, j]
            ],
            ["Enfermos", "Sanos"],
            "Covarianzas"
        )

        axes[0, 0].set_ylabel("Valor")
        axes[1, 0].set_ylabel("Valor")
        axes[1, 1].set_ylabel("Covarianza")

        axes[0, 0].set_ylim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].axhline(0, color="gray", linewidth=0.8)

        axes[0, 0].legend()

        fig.suptitle(nombre_par)
        plt.tight_layout()
        plt.show()

    graficar_resultados(resumen_12, [0, 1], "Tests 1 y 2")
    graficar_resultados(resumen_13, [0, 2], "Tests 1 y 3")

if __name__ == "__main__":
    freeze_support()
    main()

#Conclusiones:
#El modelo converge correctamente, se observa un r_hat correcto en los parametros y las muestras efectivas (aunque podrian ser más) son suficientes...

#En General:
#Las Prevalencias siguen siendo subestimadas.
#La estimación de Sensibilidades mejoraron respecto al modelo 1, pero se siguen sobreestimando.
#La estimación de Especificidades no se encuentran tan lejos de los valores verdaderos.
#Las covarianzas no se estiman con precisión, aunque su intervalo contiene el valor verdadero.

#En particular, el modelo aplicado en los test 1 y 3 estiman correctamente todos los parámetros.