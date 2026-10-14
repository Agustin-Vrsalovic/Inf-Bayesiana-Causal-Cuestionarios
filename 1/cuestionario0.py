# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: global
#     language: python
#     name: python3
# ---

# %%
import math
import random
import inspect
import warnings

respuestas = {}

# %% [markdown]
# # 1 - Preguntas sobre fundamentos
#
# Este notebook contiene una lista de preguntas junto con una lista exhaustiva de respuestas mutuamente contradictorias. 
# A diferencia de los enunciados de tipo "multiple choise" en los que se pide seleccionar una única opción, aquí se pide que distribuyan creencias entre las diferentes opciones, asegurándose que el valor asignado sea positivo y la suma sea 1.
# La evaluación será el producto de las creencias asiganadas a las respuestas correctas. 
# En caso de que la respuesta sea una variable aleatoria, se considerará la predicción típica a largo plazo, es decir, su media geométrica.
# Notar que un único cero en la secuencia anula todo el producto.
# Por ello, en caso de duda, no conviene que concentren toda su creencia en una sola opción, sino distribuir algo de creencia en todas las opciones que consideran posibles.
# Notar también que conviene asignar más a la opción en la que más creen, porque distribuir creencias en partes iguales entre todas las opciones no es mucho mejor que el azar (baseline).

# %% [markdown]
# ### Moneda
#
# ¿Cuál será el resultado del lanzamiento de una moneda?
#
# 0. Anverso (Cara)
# 0. Reverso (Sello)
# 0. Canto (Borde)

# %%
respuestas["Moneda"] = [
0.49, # 0. Anverso (Cara)
0.49, # 1. Reverso (Sello)
0.02, # 2. Canto (Borde)
"""
Justifique
Si el lanzamiento de moneda es simulado y solo hay dos opciones equiprobables. es de esperarse que salga cara o sello (el borde no seria una opcion). en cambio ,
si la moneda se lanza en la realidad y esta calibrada, existe una infima probabilidad de que salga el borde, de lo contrario saldria la cara o el sello con la misma probabilidad,
en la mayoria de los casos.
""",
]

# %% [markdown]
# ### 1.1 Cajas
#
# Hay tres cajas idénticas. Sabemos que detrás de una de ellas hay un regalo. El resto están vacías. ¿Dónde está el regalo?
#
# 0. Caja 0
# 1. Caja 1
# 2. Caja 2
# 3. Otro lugar

# %%
respuestas["Cajas"] = [
0.33, # 0. Caja 0
0.33, # 1. Caja 1
0.33, # 2. Caja 2
0.01, # 3. Otro lugar
"""
Justifique
No tenemos informacion previa sobre en que caja puede estar el regalo. Lo que si sabemos es que NO está en Otro Lugar. Por lo que se reparte equiproblamente entre las 3 cajas
""",
]

# %% [markdown]
# ### Mentir
#
# ¿Cuál de todas las opciones se considera una definición matemática del principio de no mentir?
#
# 0. Máxima incertidumbre (o entropía)
# 0. Mínima incertidumbre (o entropía)
# 0. Máxima incertidumbre (o entropía) dada la información disponible (restricciones)
# 0. Mínima incertidumbre (o entropía) dada la información disponible (restricciones)
# 0. Ninguna de las anteriores

# %%
respuestas["Mentir"] = [
0.05, # 0. Máxima incertidumbre (entropía)
0.05, # 1. Mínima incertidumbre (entropía)
0.7, # 2. Máxima incertidumbre (entropía) dada la información disponible (restricciones)
0.1, # 3. Mínima incertidumbre (entropía) dada la información disponible (restricciones)
0.1, # 4. Ninguna de las anteriores
"""
Justifique
Cuando disponemos de poca informacion, partimos de el caso de mayor incertidumbre, limitado por los datos que si se conozcan del experimento.
De esta manera no incorporamos supuestos que no estén respaldados por la información disponible.
""",
]

# %% [markdown]
# ### Universos
#
# En contextos de incertidumbre los posibles universos se bifurcan.
# Supongamos que hay tres cajas idénticas y detrás de una de ellas hay un regalo (el resto van a quedar vacías).
# Supongamos que nos permiten reservar una caja y luego, una persona nos muestra que en una de las otras cajas no está el regalo.
# En el contexto en el que reservamos la caja 1: ¿Cuál de todos los universos mutuamente contradictorios va a ocurrir?
# ¿El regalo está en la caja 1 y nos muestran la caja 1? ¿El regalo está en la caja 1 y nos muestran la caja 2?
# ... ¿El regalo está en la caja 3 y nos muestran la caja 2? ¿El regalo está en la caja 3 y nos muestran la caja 3?
#
# 0. Regalo = 1, Abren = 1
# 1. Regalo = 1, Abren = 2
# 2. Regalo = 1, Abren = 3
# 3. Regalo = 2, Abren = 1
# 4. Regalo = 2, Abren = 2
# 5. Regalo = 2, Abren = 3
# 6. Regalo = 3, Abren = 1
# 7. Regalo = 3, Abren = 2
# 8. Regalo = 3, Abren = 3
#

# %%
respuestas["Universos"] = [
0.02, # 0. Regalo = 1, Abren = 1
0.15,  # 1. Regalo = 1, Abren = 2
0.15,  # 2. Regalo = 1, Abren = 3
0.02, # 3. Regalo = 2, Abren = 1
0.02, # 4. Regalo = 2, Abren = 2
0.3,  # 5. Regalo = 2, Abren = 3
0.02, # 6. Regalo = 3, Abren = 1
0.3, # 7. Regalo = 3, Abren = 2
0.02, # 8. Regalo = 3, Abren = 3
"""
Justifique
Es el conocido problema de Monty Hall. Es probable que el presentador abra una de las dos cajas que NO elegiste. Entre ellas, es más probable que el premio se encuentre en la
caja que no abrio (lo dice la consigna). Por ultimo, es mas probable que la caja NO elegida tenga el Regalo, debido a que el presentador si sabe que puerta tiene el premio, por
lo que modifica el resultado final.
""",
]


# %% [markdown]
# ### Historia
#
# En los últimos siglos hubieron muchos avances científicos.
# En los últimos años, en particular, se han producido enormes avances en el área de la inteligencia artificial.
# ¿Cuándo ocurrió el primer uso conocido del actual sistema de razonamiento para contextos de incertidumbre?
#
# 0. Siglo 21
# 1. Siglo 20
# 2. Siglo 19
# 3. Siglo 18
# 4. Siglo 17
# 5. Antes

# %%
respuestas["Historia"] = [
0.898, # 0. Siglo 21
0.05, # 1. Siglo 20
0.03, # 2. Siglo 19
0.02, # 3. Siglo 18
0.001, # 4. Siglo 17
0.001, # 5. Antes
"""
Justifique
Dado el grado exponencial del avance de la tecnologia y la ciencia en la historia, considero que es mas probable que sea un hecho reciente. Ademas, en el sigro 21 se
abarataron mucho los costos computacionales, los cuales son excelentes para crear un sistema de razonamiento.
""",
]


# %% [markdown]
# ### Conjunta
#
# La distribución de creencias conjunta sobre varias variables se puede descomponer como el producto de la probabilidad marginal de cualquiera de las variables,
# multiplicado por la probabilidad condicional del resto de las variables dada la primera variable (usada en la marginal).
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Conjunta"] = [
0.8, # 0. Siempre
0.05, # 1. A veces
0.15, # 2. Nunca
"""
Justifique
La afirmación es correcta y se cumple en todas las ocaciones. Viene de la regla del producto de probabilidades, y no depende de que las variables sean independientes.
""",
]


# %% [markdown]
# ### Independencia
#
# Si A es independiente de B sabemos que P(A|B) = P(B). ¿Pero es cierta la siguiente igualdad: P(A)P(B|A) = P(A)P(B)?
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Independencia"] = [
0.8, # 0. Siempre
0.05, # 1. A veces
0.15, # 2. Nunca
"""
Justifique
Siempre se cumple. Si A y B son independientes, saber que ocurrió "A" no cambia la probabilidad de "B".
Entonces P(B|A) es igual a P(B), y se mantiene la igualdad.
""",
]


# %% [markdown]
# ### Descomposiciones
#
# Sabemos que siempre existe alguna forma de descomponer la distribución conjunta como el producto de distribuciones condicionales unidimensionales.
# ¿Si hay N variables, cuántas descomposiciones en total existen?
#
# 0. 1
# 1. N - 1
# 2. N
# 3. N * (N - 1)
# 4. N * N
# 5. N! - 1
# 6. N!
# 7. N ^ N - 1
# 8. N ^ N

# %%
respuestas["Descomposiciones"] = [
0.0125, # 0. 1
0.0125, # 1. N - 1
0.0125, # 2. N
0.0125, # 3. N * (N - 1)
0.0125, # 4. N * N
0.0125, # 5. N! - 1
0.9, # 6. N!
0.0125, # 7. N ^ N - 1
0.0125, # 8. N ^ N
"""
Justifique
Hay N! formas porque cada descomposición depende del orden en que elegimos las variables para aplicar la regla de la cadena.
Existen N formas de elegir la primera variable, luego N-1 para la segunda, ...
Si multiplicas las opciones en cada posición del ordenamiento, el total es N!
""",
]


# %% [markdown]
# ### Teorema de Bayes.
#
# El teorema de Bayes nos permite actualizar las creencias de las hipótesis internas a los modelos causales dado los datos.
# El denominador del teorema de Bayes es constante para las diferentes hipótesis.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Teorema de Bayes"] = [
0.2, # 0. Siempre
0.7, # 1. A veces
0.1, # 2. Nunca
"""
Justifique
El denominador de Bayes es constante cuando se comparan distintas hipótesis usando los mismos datos observados.
Si cambian los datos o el conjunto de comparación, puede dejar de ser constante.
""",
]



# %% [markdown]
# ### Predicciones
#
# La predicción que una hipótesis H hace de un conjunto de datos, P(Datos = {d1, ..., d_n} | H), puede calcularse como el producto de las predicciones que la hipótesis hace de cada dato individual dado los datos ya vistos, P(d1|H)P(d2|d1,H)...
# Para que el cálculo sea correcto es importante que se respete el orden en el cual los datos fueron observados en los hechos, es decir, que no ocurra P(d2|H)P(d1|d2,H)..
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Predicciones"] = [
0.15, # 0. Siempre
0.05, # 1. A veces
0.8, # 2. Nunca
"""
Justifique
No es necesario respetar el orden en que se observaron los datos, ya que la regla de la cadena permite descomponer la probabilidad conjunta usando cualquier orden.
Mientras se mantengan bien definidas las probabilidades condicionales, P(d1|H)P(d2|d1,H) y P(d2|H)P(d1|d2,H) dan el mismo resultado.
""",
]


# %% [markdown]
# ### Valor de verdad
#
# Si una hipótesis predice con 0 uno de los datos observados, la hipótesis se hace falsa.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Valor de verdad"] = [
0.8, # 0. Siempre
0.05, # 1. A veces
0.15, # 2. Nunca
"""
Justifique
Si P(d∣H)=0, entonces la Hipotesis siempre quedara descartada, debido a que es una situacion imposible
""",
]


# %% [markdown]
# ### Teorías causales
#
# Históricamente todas las ciencias con datos, desde la física hasta las ciencias sociales, explicaron el mundo a través de teorías causales.
# Los recientes avances en el área de aprendizaje automático e inteligencia artificial, sin embargo, se produjeron por el desarrollo de algoritmos altamente predictivos sin ninguna interpretación causal.
# ¿Qué relación hay entre los modelos causal y los complejos algoritmos de AI/ML?
#
# 0. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
# 1. El modelo causal que se corresponde con la realidad causal subyacente a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
# 2. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
# 3. No son comparables porque los modelos causales solo explican, no predicen.
# 4. Ninguna de las anteriores

# %%
respuestas["Teorías causales"] = [
0.05, # 0. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
0.05, # 1. El modelo causal que se corresponde con la realidad causal subyacente a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
0.7, # 2. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
0.05, # 3. No son comparables porque los modelos causales solo explican, no predicen.
0.15, # 4. Ninguna de las anteriores
"""
Justifique
Un modelo causal representa correctamente la realidad subyacente, contiene toda la información necesaria para explicar y predecir los datos.
Un algoritmo de IA puede igualar ese desempeño, pero no debería superarlo usando la misma información.
""",
]


# %% [markdown]
# ### Predicción e información
#
# Cuanto mejor se predice más información (de Shannon) se obtiene.
#
# 0. Siempre
# 1. A veces
# 2. Nunca


# %%
respuestas["Predicción e información"] = [
0.05, # 0. Siempre
0.15, # 1. A veces
0.8, # 2. Nunca
"""
Justifique
En la teoría de la información de Shannon cuanto más predecible es un evento, menos información aporta cuando ocurre.
""",
]


# %% [markdown]
# ### Modelos e información
#
# Al evaluar modelos causales, preferimos el que acumula más información (de Shannon).
#
# 0. Siempre
# 1. A veces
# 2. Nunca


# %%
respuestas["Modelos e información"] = [
0.8, # 0. Siempre
0.15, # 1. A veces
0.05, # 2. Nunca
"""
Justifique
Al evaluar modelos se prefiere el que asigna mayor probabilidad a los datos observados.
Eso equivale a minimizar la sorpresa, es decir, maximiza la información que el modelo explica sobre los datos.
""",
]

# %% [markdown]
# ### Evaluación de modelos
#
# Podemos identificar el modelo causal correcto si observamos suficientes datos.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Evaluación de modelos"] = [
0.05, # 0. Siempre
0.4, # 1. A veces
0.55, # 2. Nunca
"""
Justifique
Teniendo datos observacionales no siempre se puede identificar el modelo causal verdadero.
Distintos modelos causales pueden generar las mismas distribuciones de datos, por lo que hacen falta supuestos adicionales o intervenciones para distinguirlos.
""",
]

# %% [markdown]
# ### Contrafactuales
#
# Cuando conocemos los mecasnimos causales probabilísticos de cada variable podemos usar la información factual para predecir cuál hubiera sido un resultado contrafactual.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Contrafactuales"] = [
0.8, # 0. Siempre
0.15, # 1. A veces
0.05, # 2. Nunca
"""
Justifique
Si conocemos los mecanismos causales probabilísticos de cada variable, podemos combinar esa información con los datos observados
para estimar qué habría pasado si las condiciones hubieran sido distintas.
""",
]



# %% [markdown]
# ### Diversificación
#
# Una casa de apuestas nos paga 3 por Cara y 1.2 por Sello por el lanzamiento de moneda.
# La moneda es normal, con 0.5 de probabilidad de que salga Cara o Sello.
# Supongamos que nos ofrecen jugar 10000 veces, pero apostando absolutamente todos los recursos en cada paso temporal.
# Apostamos todo, nos devuelven actualizado y volvemos a apostar.
# ¿Qué proporción conviene apostar a Cara?
# Notar que el resto se asigna a Sello.
# Notar además que si apostamos todo a Cara y sale Sello perdemos todos los recursos y no podemos volver a jugar (solo nos pagan en el lado donde sale la moneda).
#
# 0. Recursos asignados a Cara: 0.0
# 1. Recursos asignados a Cara: 0.1
# 2. Recursos asignados a Cara: 0.2    
# 3. Recursos asignados a Cara: 0.3
# 4. Recursos asignados a Cara: 0.4
# 5. Recursos asignados a Cara: 0.5
# 6. Recursos asignados a Cara: 0.6
# 7. Recursos asignados a Cara: 0.7
# 8. Recursos asignados a Cara: 0.8
# 9. Recursos asignados a Cara: 0.9
# 10. Recursos asignados a Cara: 1.0

# %%
respuestas["Diversificación"] = [
0.01, # 0. Recursos asignados a Cara: 0.0
0.015, # 1. Recursos asignados a Cara: 0.1
0.02, # 2. Recursos asignados a Cara: 0.2
0.025, # 3. Recursos asignados a Cara: 0.3
0.03, # 4. Recursos asignados a Cara: 0.4
0.8,  # 5. Recursos asignados a Cara: 0.5
0.03, # 6. Recursos asignados a Cara: 0.6
0.025, # 7. Recursos asignados a Cara: 0.7
0.02, # 8. Recursos asignados a Cara: 0.8
0.015, # 9. Recursos asignados a Cara: 0.9
0.01, # 10. Recursos asignados a Cara: 1.0
"""
Justifique
Por mas que uno crea que conviene apostarle mas a la Cara, como ambas tienen la misma probabilidad; la mejor estrategia a largo plazo es apostarle el 0.5 a cada lado de la moneda
""",
]


# %% [markdown]
# ### Apuesta individual
#
# Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello.
# Nos ofrecen jugar 10000 veces, apostando en cada ocasión todos nuestros recursos, 50% a Cara y 50% a Sello.
# ¿Nos conviene jugar?
# Notar que cuando sale Cara, crecen nuestros recursos 50% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 3*50=150).
# Notar que cuando sale Sello, crecen nuestros recursos -40% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 1.2*50=60).
#
# 0. No
# 1. Sí

# %%
respuestas["Apuesta individual"] = [
0.1, # 0. No
0.9, # 1. Sí
"""
Justifique
Como vimos antes, a largo plazo conviene la estrategia del 50 | 50, por un tema de probabilidades y disminuir el riesgo de perder todos los recursos.
""",
]


# %% [markdown]
# ### Teoría de utilidad esperada
#
# La teoría de utilidad esperada dice que debemos aceptar una apuesta cuando la utilidad esperada es positiva.
# Supongamos que elegimos una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello (el caso anterior).
# Notar que la esperanza de los recursos es positiva, crece a 5% por paso temporal ((150+60)/2=210/2=105).
# Supongamos que nos ofrecen jugar 10000 veces en un instante, usando absolutamente todos los recursos.
# ¿No conviene jugar?
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Teoría de Utilidad Esperada"] = [
0.8, # 0. Siempre
0.15, # 1. A veces
0.05, # 2. Nunca
"""
Justifique
Como ya vimos y lo explica la teoria de la utilidad esperada; si la esperanza de los recursos es positiva, si conviene apostar el 50 | 50.
""",
]


# %% [markdown]
# ### Fondo común
#
# Supongamos que ya estamos jugando una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello.
# ¿Nos conviene juntarnos con alguien, y al final de cada paso temporal poner todos los recursos en un fondo común y dividirlos en partes iguales?
#
#
# 0. No conviene
# 1. Indistinto
# 2. Sí conviene

# %%
respuestas["Fondo común"] = [
0.05, # 0. No conviene
0.05, # 1. Indistinto
0.9, # 2. Sí conviene
"""
Justifique
Si, conviene. Ya que al aumentar la cantidad de personas, disminuis la incertidumbre, por lo tante disminuye el riesgo.
Como la esperanza es positiva, a largo plazo se espera ganar.
""",
]


# %% [markdown]
# ### Impuestos
#
# Supongamos que ya estamos jugando una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello, y ya estamos hace mucho tiempo jugando con un fondo común de en un grupo de 100 personas.
# ¿Qué pasa con la tasa de crecimiento de nuestros recursos si logramos encontrar la forma de dejar de aportar nuestra cuota al fondo común mientras tenemos garantizado que seguimos recibimos la cuota del fondo en partes iguales?
#
# 0. Disminuye
# 1. No cambia
# 2. Aumenta

# %%
respuestas["Impuestos"] = [
0.1, # 0. Disminuye
0.1, # 1. No cambia
0.8, # 2. Aumenta
"""
Justifique
Aumenta porque al dejar de aportar al fondo común seguimos recibiendo los beneficios del crecimiento colectivo, pero nuestro aporte deja de reducir nuestro capital disponible.
Al tener menos Capital arriesgado y se mantienen las ganancias, esto hace que la tasa de crecimiento efectiva de nuestros recursos sea mayor.
""",
]