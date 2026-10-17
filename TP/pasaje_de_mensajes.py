import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.special import betaln, gammaln
#Para el gráfico final
from scipy.stats import beta as beta_dist

from matplotlib.lines import Line2D

# Datos: conversiones (n) y tamaño (N) de cada grupo, por mes
data = pd.read_csv("TP/CSVs/01-Incrementalidad.csv", encoding="utf-8-sig")
def serie(tipo):
    return data[data["Tipo"] == tipo]["online_schedules"].to_numpy()

meses = data["Mes"].unique()
T = len(meses)
n_C, N_C = serie("Eventos CG"), serie("Usuarios CG")
n_T, N_T = serie("Eventos Tratamiento"), serie("Usuarios Tratamiento")

def calcular_mensajes(n, N, lam, prior=(1, 20)):
    #T = len(n)

    #Datos totales observados en cada mes
    likelihood = np.column_stack((n, N - n))

    #Mensajes para el recorrido hacia adelante (forward)
    forward_prior = np.zeros((T, 2))
    forward_post = np.zeros((T, 2))

    #Usamos el prior asignado para el primer mensaje
    forward_prior[0] = prior

    for t in range(T):
        if t > 0:
            forward_prior[t] = lam * forward_post[t - 1]

        forward_post[t] = forward_prior[t] + likelihood[t]

    #Mensajes para el recorrido hacia atrás (backward)
    backward_prior = np.zeros((T, 2))
    backward_post = np.zeros((T, 2))

    for t in range(T - 1, -1, -1):
        if t < T - 1:
            backward_prior[t] = lam * backward_post[t + 1]

        backward_post[t] = backward_prior[t] + likelihood[t]

    #Combinamos la información de los meses vecinos, sin usar la del mes a predecir
    within_prior = forward_prior + backward_prior

    #Incorporamos los datos observados del mes, para obtener la posterior
    posterior = within_prior + likelihood

    return {
        "likelihood": likelihood,
        "forward_prior": forward_prior,
        "forward_post": forward_post,
        "backward_prior": backward_prior,
        "backward_post": backward_post,
        "within_prior": within_prior,
        "posterior": posterior
    }

#Primera prueba para testear el algoritmo, con un lambda fijo
mensajes_C = calcular_mensajes(n_C, N_C, lam=0.5)

print("\nWithinPrior de enero:")
print(mensajes_C["within_prior"][0])

print("\nPosterior aproximado de enero:")
print(mensajes_C["posterior"][0])

#La variable Lambda controla cuánta información se mantiene al transmitirla de un mes al siguiente
#Seleccion de Lambda ideal para este modelo.

def calcular_log_evidencia(n, N, within_prior):

    #Se calculan las log-evidencias para todos los meses con diferentes Lambdas y comparamos los resultados para elegir al mejor
    #Usamos within_prior para que el modelo haga las predicciones sin tener informacion previa sobre los resultados del mes que se quieren predecir

    n = np.asarray(n)
    N = np.asarray(N)

    #Extraigo las primeras columnas
    alpha = within_prior[:, 0]
    beta = within_prior[:, 1]

    #Logaritmo del coeficiente binomial
    log_coef = (gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1))

    #Log-evidencia predictiva de cada mes
    log_evidencias = (log_coef + betaln(alpha + n, beta + N - n) - betaln(alpha, beta)
    )

    #Una sola puntuación para toda la serie
    return log_evidencias.sum()


def evaluar_lambdas(n, N, valores_lambda):

    resultados = []

    #Pruebo cada valor candidato de lambda
    for lam in valores_lambda:

        #Se calculan los mensajes para toda la serie
        mensajes = calcular_mensajes(n, N, lam)

        #Uso como prior sin los datos locales, del mes a predecir
        within_prior = mensajes["within_prior"]

        log_evidencia = calcular_log_evidencia(n, N, within_prior)

        #Guardamos lambda y su puntuacion
        resultados.append({"lambda": lam,
                           "log_evidencia": log_evidencia})

    #Formato Pandas
    resultados = pd.DataFrame(resultados)

    #Le resto max a cada Log_Evidencia. El mejor valor tiene puntuacion relativa igual a cero
    resultados["evidencia_relativa"] = (resultados["log_evidencia"] - resultados["log_evidencia"].max())

    return resultados

#Primera Busqueda de Valores candidatos de lambda. Con valores entre 0.001 y 1
valores_lambda = np.linspace(0.001, 1.0, 300)

#Evaluamos las cadenas de control y tratamiento por separado
resultados_C = evaluar_lambdas(n_C, N_C, valores_lambda)
resultados_T = evaluar_lambdas(n_T, N_T, valores_lambda)

#Seleccion del lambda con mayor evidencia.
mejor_C = resultados_C.loc[resultados_C["log_evidencia"].idxmax()]
mejor_T = resultados_T.loc[resultados_T["log_evidencia"].idxmax()]

print("1er Busqueda de Lambda | Mejores Valores:")
print(f"Control:     lambda = {mejor_C['lambda']:.4f}")
print(f"Tratamiento: lambda = {mejor_T['lambda']:.4f}")

#Graficamos evidencia relativa frente a lambda
plt.figure(figsize=(9, 5))

plt.plot(
    resultados_C["lambda"],
    resultados_C["evidencia_relativa"],
    label="Control"
)

plt.plot(
    resultados_T["lambda"],
    resultados_T["evidencia_relativa"],
    label="Tratamiento"
)

plt.axhline(0, linestyle=":", linewidth=1)
plt.xlabel("Lambda")
plt.ylabel("Log-evidencia relativa al máximo")
plt.title("Selección de lambda mediante evidencia predictiva")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

#Se observan que ambos lambas (uno por grupo) tienen valores muy distintos; conviene tratarlos por separado
#Probe por separado con un lamda único para ambos grupos y no es la mejor opción
#Ambos tienen su pico máximo en un rango de valores menor a 0.4

#Segunda Busqueda de valores candidatos de lambda. Reajustamos la busqueda entre 0 y 0.4
valores_C = np.linspace(0.001, 0.4, 2000)
valores_T = np.linspace(0.001, 0.05, 2000)

resultados_C_finos = evaluar_lambdas(n_C, N_C, valores_C)
resultados_T_finos = evaluar_lambdas(n_T, N_T, valores_T)

mejor_C = resultados_C_finos.loc[resultados_C_finos["log_evidencia"].idxmax()]
mejor_T = resultados_T_finos.loc[resultados_T_finos["log_evidencia"].idxmax()]

print("2da Busqueda de Lambda | Mejores Valores:")
print(f"Control:     {mejor_C['lambda']:.5f}")
print(f"Tratamiento: {mejor_T['lambda']:.5f}")

# Graficamos cada busqueda con mayor detalle.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, resultados, nombre in [
    (axes[0], resultados_C_finos, "Control"),
    (axes[1], resultados_T_finos, "Tratamiento")
]:
    ax.plot(
        resultados["lambda"],
        resultados["evidencia_relativa"]
    )
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Log-evidencia relativa")
    ax.set_title(nombre)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

#Reajustamos el modelo para los nuevos valores de Lambda por grupo:
#Parámetros elegidos:
LAMBDA_CONTROL = 0.17549
LAMBDA_TRATAMIENTO = 0.01541
PRIOR_INICIAL = (1.0, 20.0)

#Ejecutamos forward-backward para cada grupo
mensajes_C = calcular_mensajes(n_C, N_C, lam=LAMBDA_CONTROL, prior=PRIOR_INICIAL)
mensajes_T = calcular_mensajes(n_T, N_T, lam=LAMBDA_TRATAMIENTO, prior=PRIOR_INICIAL)

#Extraemos los parámetros de la Beta del posterior de cada mes
alpha_C, beta_C = mensajes_C["posterior"].T
alpha_T, beta_T = mensajes_T["posterior"].T

#Medias posteriores
media_C = alpha_C / (alpha_C + beta_C)
media_T = alpha_T / (alpha_T + beta_T)

# Intervalos de credibilidad centrales del 95 %
ic_C_inf = beta_dist.ppf(0.025, alpha_C, beta_C)
ic_C_sup = beta_dist.ppf(0.975, alpha_C, beta_C)

ic_T_inf = beta_dist.ppf(0.025, alpha_T, beta_T)
ic_T_sup = beta_dist.ppf(0.975, alpha_T, beta_T)

# Referencia independiente: cada mes usa solamente sus propios datos
alpha_C_ind = 1 + n_C
beta_C_ind = 20 + N_C - n_C

alpha_T_ind = 1 + n_T
beta_T_ind = 20 + N_T - n_T

# Probabilidad de que el tratamiento supere al control
# Mediante muestras de los posteriores Beta, sin usar MCMC
rng = np.random.default_rng(111)
N_SIM = 200000

p_temporal = []
p_independiente = []

for t in range(len(meses)):
    # Posteriores del modelo temporal
    p = rng.beta(alpha_C[t], beta_C[t], size=N_SIM)
    q = rng.beta(alpha_T[t], beta_T[t], size=N_SIM)
    p_temporal.append(np.mean(q > p))

    # Posteriores independientes por mes
    p0 = rng.beta(alpha_C_ind[t], beta_C_ind[t], size=N_SIM)
    q0 = rng.beta(alpha_T_ind[t], beta_T_ind[t], size=N_SIM)
    p_independiente.append(np.mean(q0 > p0))

p_temporal = np.array(p_temporal)
p_independiente = np.array(p_independiente)

# Tabla resumen
resultados = pd.DataFrame({
    "Mes": meses,
    "Control observado (%)": 100 * n_C / N_C,
    "Media control (%)": 100 * media_C,
    "IC95 control inferior (%)": 100 * ic_C_inf,
    "IC95 control superior (%)": 100 * ic_C_sup,
    "Tratamiento observado (%)": 100 * n_T / N_T,
    "Media tratamiento (%)": 100 * media_T,
    "IC95 tratamiento inferior (%)": 100 * ic_T_inf,
    "IC95 tratamiento superior (%)": 100 * ic_T_sup,
    "P(q > p) temporal": p_temporal,
    "P(q > p) independiente": p_independiente
})

print(f"Lambda control: {LAMBDA_CONTROL:.5f}")
print(f"Lambda tratamiento: {LAMBDA_TRATAMIENTO:.5f}")
print("\nRESULTADOS POR MES")
print(resultados.round(4).to_string(index=False))

resultados.to_csv(
    "incrementalidad_forward_backward_resultados.csv",
    index=False
)

# Gráficos
x = np.arange(len(meses))

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 9), sharex=True)

# Grafico del desempeño observado y posterior temporal.
for nombre, color, media, inferior, superior, observada in [
    ("Control", "C0", media_C, ic_C_inf, ic_C_sup, n_C / N_C),
    ("Tratamiento", "C1", media_T, ic_T_inf, ic_T_sup, n_T / N_T)
]:
    ax1.errorbar(
        x,
        100 * media,
        yerr=[
            100 * (media - inferior),
            100 * (superior - media)
        ],
        color=color,
        ecolor=color,
        marker="o",
        capsize=3,
        label=f"{nombre}: posterior e IC 95 %")

    ax1.scatter(
        x,
        100 * observada,
        color=color,
        marker="x",
        label=f"{nombre}: observado")

ax1.set_ylabel("Conversiones (%)")
ax1.set_title("Desempeño mensual: observado y posterior temporal")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.25)

#Probabilidad (q > p)
ax2.plot(
    x, p_temporal, marker="o",
    label="Forward-backward temporal")

ax2.plot(
    x, p_independiente, marker="s", linestyle="--",
    label="Meses independientes")

ax2.axhline(
    0.95, linestyle=":",
    label="Referencia 0.95")

ax2.set_xticks(x, [str(m) for m in meses], rotation=45)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("Mes")
ax2.set_ylabel("P(q > p | datos)")
ax2.set_title("Probabilidad de que el tratamiento supere al control")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

fig.tight_layout()
plt.savefig(
    "incrementalidad_forward_backward_resultados.png",
    dpi=160,
    bbox_inches="tight")
plt.show()

#Gráfico de la distribución conjunta de control y tratamiento.

def graficar_conjuntas(alpha_control, beta_control,
                       alpha_tratamiento, beta_tratamiento, titulo):

    valores = np.linspace(0.0025, 0.0065, 250)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for t in range(len(meses)):

        # Densidad posterior de control y tratamiento.
        densidad_p = beta_dist.pdf(
            valores, alpha_control[t], beta_control[t]
        )

        densidad_q = beta_dist.pdf(
            valores, alpha_tratamiento[t], beta_tratamiento[t]
        )

        # Producto de las dos densidades:
        # filas = tratamiento, columnas = control.
        conjunta = np.outer(densidad_q, densidad_p)

        ax = axes[t]

        # Graficamos la distribucion conjunta.
        ax.contourf(
            valores * 100,
            valores * 100,
            conjunta,
            levels=20
        )

        # Diagonal q = p.
        ax.plot(
            valores * 100,
            valores * 100,
            "k--",
            linewidth=1.5
        )

        ax.set_title(str(meses[t]))
        ax.set_xlim(0.25, 0.65)
        ax.set_ylim(0.25, 0.65)
        ax.set_aspect("equal")
        ax.set_xlabel("Control (%)")
        ax.set_ylabel("Tratamiento (%)")

    fig.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.show()


#Modelo temporal:
graficar_conjuntas(
    alpha_C, beta_C, alpha_T, beta_T,
    "Distribuciones conjuntas - Modelo temporal")

#Modelo independiente:
graficar_conjuntas(
    alpha_C_ind, beta_C_ind, alpha_T_ind, beta_T_ind,
    "Distribuciones conjuntas - Modelo independiente")