# Devolución — Cuestionario 0

**Estudiante:** VRSALOVIC, AGUSTIN
**Divergencia KL global:** 29.94

---

### Moneda

- Divergencia KL: 0.0303

**Tu respuesta:** [0.49, 0.49, 0.02]
> Justifique
Si el lanzamiento de moneda es simulado y solo hay dos opciones equiprobables. es de esperarse que salga cara o sello (el borde no seria una opcion). en cambio ,
si la moneda se lanza en la realidad y esta calibrada, existe una infima probabilidad de que salga el borde, de lo contrario saldria la cara o el sello con la misma probabilidad,
en la mayoria de los casos.

**Respuesta de referencia:** [0.48, 0.52, 2e-07]
> Una moneda normal tiene dos caras, por lo que la probabilidad de que salga Anverso o Reverso es aproximadamente 0.5 cada una. Una vez cada 5 millones de tiradas queda en el borde.

### Cajas

- Divergencia KL: 1.599

**Tu respuesta:** [0.33, 0.33, 0.33, 0.01]
> Justifique
No tenemos informacion previa sobre en que caja puede estar el regalo. Lo que si sabemos es que NO está en Otro Lugar. Por lo que se reparte equiproblamente entre las 3 cajas

**Respuesta de referencia:** [0, 0, 1, 0]
> En este caso, por casualidad, el regalo se encontraba en la tercera caja (índice 2).

### Mentir

- Divergencia KL: 0.5146

**Tu respuesta:** [0.05, 0.05, 0.7, 0.1, 0.1]
> Justifique
Cuando disponemos de poca informacion, partimos de el caso de mayor incertidumbre, limitado por los datos que si se conozcan del experimento.
De esta manera no incorporamos supuestos que no estén respaldados por la información disponible.

**Respuesta de referencia:** [0, 0, 1, 0, 0]
> Maximizar incertidumbre (entropía) dada la información disponible (restricciones) garantiza no mentir: decir que sabemos cuando no sabemos y decir que no sabemos cuando sí sabemos.

### Universos

- Divergencia KL: 0.152

**Tu respuesta:** [0.02, 0.15, 0.15, 0.02, 0.02, 0.3, 0.02, 0.3, 0.02]
> Justifique
Es el conocido problema de Monty Hall. Es probable que el presentador abra una de las dos cajas que NO elegiste. Entre ellas, es más probable que el premio se encuentre en la
caja que no abrio (lo dice la consigna). Por ultimo, es mas probable que la caja NO elegida tenga el Regalo, debido a que el presentador si sabe que puerta tiene el premio, por
lo que modifica el resultado final.

**Respuesta de referencia:** [0, 0.1667, 0.1667, 0, 0, 0.3333, 0, 0.3333, 0]
> La realidad causal podría tener cualquier distribución que tenga 0 en los casos imposibles (Abren=1 es imposible porque la caja 1 está reservada; Regalo=1,Abren=1 y Regalo=2,Abren=2 y Regalo=3,Abren=3 son imposibles porque no se abre la caja con el regalo).
En este caso particular la referencia propone una realidad causal subyacente que tiene la misma distribución que la predicción de máxima incertidumbre (entropía) dada la información disponible (restricciones).

### Historia

- Divergencia KL: 5.644

**Tu respuesta:** [0.898, 0.05, 0.03, 0.02, 0.001, 0.001]
> Justifique
Dado el grado exponencial del avance de la tecnologia y la ciencia en la historia, considero que es mas probable que sea un hecho reciente. Ademas, en el sigro 21 se
abarataron mucho los costos computacionales, los cuales son excelentes para crear un sistema de razonamiento.

**Respuesta de referencia:** [0, 0, 0, 1, 0, 0]
> Según nuestro conocimiento, el primer uso fue en el siglo 18, por el señor Bayes.

### Conjunta

- Divergencia KL: 0.3219

**Tu respuesta:** [0.8, 0.05, 0.15]
> Justifique
La afirmación es correcta y se cumple en todas las ocaciones. Viene de la regla del producto de probabilidades, y no depende de que las variables sean independientes.

**Respuesta de referencia:** [1, 0, 0]
> Por la regla de la cadena de la probabilidad, cualquier distribución conjunta siempre puede factorizarse de manera exacta como el producto de una marginal y la condicional del resto de variables dadas las anteriores. Es una identidad algebraica universal.

### Independencia

- Divergencia KL: 0.3219

**Tu respuesta:** [0.8, 0.05, 0.15]
> Justifique
Siempre se cumple. Si A y B son independientes, saber que ocurrió "A" no cambia la probabilidad de "B".
Entonces P(B|A) es igual a P(B), y se mantiene la igualdad.

**Respuesta de referencia:** [1, 0, 0]
> Si A es independiente de B, se cumple también que P(B|A) = P(B) y por lo tanto siempre se cumple la igualdad dada en el enunciado.

### Descomposiciones

- Divergencia KL: 0.152

**Tu respuesta:** [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.9, 0.0125, 0.0125]
> Justifique
Hay N! formas porque cada descomposición depende del orden en que elegimos las variables para aplicar la regla de la cadena.
Existen N formas de elegir la primera variable, luego N-1 para la segunda, ...
Si multiplicas las opciones en cada posición del ordenamiento, el total es N!

**Respuesta de referencia:** [0, 0, 0, 0, 0, 0, 1, 0, 0]
> Cada permutación produce una factorización única utilizando la regla de la cadena, por lo que existen N! descomposiciones posibles.

### Teorema de Bayes

- Divergencia KL: 2.322

**Tu respuesta:** [0.2, 0.7, 0.1]
> Justifique
El denominador de Bayes es constante cuando se comparan distintas hipótesis usando los mismos datos observados.
Si cambian los datos o el conjunto de comparación, puede dejar de ser constante.

**Respuesta de referencia:** [1, 0, 0]
> El denominador del Teorema de Bayes es la verosimilitud marginal, P(Datos), calculada integrando todo el espacio de hipótesis.
Debido a que la hipótesis no es un parámetro de esa función, siempre es constante para las diferentes hipótesis.

### Predicciones

- Divergencia KL: 0.3219

**Tu respuesta:** [0.15, 0.05, 0.8]
> Justifique
No es necesario respetar el orden en que se observaron los datos, ya que la regla de la cadena permite descomponer la probabilidad conjunta usando cualquier orden.
Mientras se mantengan bien definidas las probabilidades condicionales, P(d1|H)P(d2|d1,H) y P(d2|H)P(d1|d2,H) dan el mismo resultado.

**Respuesta de referencia:** [0, 0, 1]
> Todas las descomposiciones generadas por la regla de la cadena son equivalentes y por lo tanto el orden temporal no juega ningún rol en el cálculo de la predicción conjunta.

### Valor de verdad

- Divergencia KL: 0.3219

**Tu respuesta:** [0.8, 0.05, 0.15]
> Justifique
Si P(d∣H)=0, entonces la Hipotesis siempre quedara descartada, debido a que es una situacion imposible

**Respuesta de referencia:** [1, 0, 0]
> Si una hipótesis asigna probabilidad nula a un dato que ha sido efectivamente observado, P(D|H)=0, por la regla de la cadena la predicción conjunta va a ser 0 siempre, P(Datos|H)=0, esa hipótesis se hace falsa para siempre, la creencia a posterior de esa hipótesis va a ser 0.

### Teorías causales

- Divergencia KL: 0.5146

**Tu respuesta:** [0.05, 0.05, 0.7, 0.05, 0.15]
> Justifique
Un modelo causal representa correctamente la realidad subyacente, contiene toda la información necesaria para explicar y predecir los datos.
Un algoritmo de IA puede igualar ese desempeño, pero no debería superarlo usando la misma información.

**Respuesta de referencia:** [0, 0, 1, 0, 0]
> La predicción que el modelo hace de los datos, P(Datos|Modelo), en escala logarítmica y en tiempo infinito (ensamble o repeticiones del proceso) es precisamente el negativo de la entropía cruzada entre el proceso generativo de los datos P(Datos|Realidad) y las predicciones P(Datos|Modelo).
Sabemos que la entropía cruzada se minimiza cuando P(Datos|Modelo) = P(Datos|Realidad).

### Predicción e información

- Divergencia KL: 0.3219

**Tu respuesta:** [0.05, 0.15, 0.8]
> Justifique
En la teoría de la información de Shannon cuanto más predecible es un evento, menos información aporta cuando ocurre.

**Respuesta de referencia:** [0, 0, 1]
> Una predicción perfecta agrega nula información.
Cuanto mejor se predice, menos información se obtiene.
Siempre.
Nunca ocurre lo contrario.

### Modelos e información

- Divergencia KL: 4.322

**Tu respuesta:** [0.8, 0.15, 0.05]
> Justifique
Al evaluar modelos se prefiere el que asigna mayor probabilidad a los datos observados.
Eso equivale a minimizar la sorpresa, es decir, maximiza la información que el modelo explica sobre los datos.

**Respuesta de referencia:** [0, 0, 1]
> Queremos el modelo causal que predice perfectamente, que no recibe ningún tipo de información de Shannon.

### Evaluación de modelos

- Divergencia KL: 1.322

**Tu respuesta:** [0.05, 0.4, 0.55]
> Justifique
Teniendo datos observacionales no siempre se puede identificar el modelo causal verdadero.
Distintos modelos causales pueden generar las mismas distribuciones de datos, por lo que hacen falta supuestos adicionales o intervenciones para distinguirlos.

**Respuesta de referencia:** [0, 1, 0]
> Es posible que dos modelos causales alternativos tengan la misma distribución conjunta.
En esos casos no podemos distinguir cuál es el modelo correcto, porque ambos hacen las mismas predicciones.

### Contrafactuales

- Divergencia KL: 0.3219

**Tu respuesta:** [0.8, 0.15, 0.05]
> Justifique
Si conocemos los mecanismos causales probabilísticos de cada variable, podemos combinar esa información con los datos observados
para estimar qué habría pasado si las condiciones hubieran sido distintas.

**Respuesta de referencia:** [1, 0, 0]
> Los contrafactuales siguen una lógica causal que puede incluirse en el modelo causal usando los mecanismos causales probabilísticos.
Ese modelo extendido contiene variables factuales y contrafactuales, y permite predecir cuál hubiera sido un resultado contrafactual dada la información factual.

### Diversificación

- Divergencia KL: 0.3219

**Tu respuesta:** [0.01, 0.015, 0.02, 0.025, 0.03, 0.8, 0.03, 0.025, 0.02, 0.015, 0.01]
> Justifique
Por mas que uno crea que conviene apostarle mas a la Cara, como ambas tienen la misma probabilidad; la mejor estrategia a largo plazo es apostarle el 0.5 a cada lado de la moneda

**Respuesta de referencia:** [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
> El proceso de actualización de los recursos tiene una estructura multiplicativa.
La variable libre es la proporción de recursos que asignamos a Cara y Sello.
Los pagos que ofrece la casa de apuestas, en cambio, no cambian.
Al comparar el cociente entre dos riquezas generadas mediante dos estrategias de diversificación alternativas, los pagos de la casa de apuestas se cancelan.
La ecuación a optimizar es la media geométrica de la diversificación dada la probabilidad de la moneda.
En escala logarítmica tiene la estructura de la entropía cruzada, con el signo invertido.
Y eso se maximiza cuando la diversificación es igual a la probabilidad de la moneda.

### Apuesta individual

- Divergencia KL: 3.322

**Tu respuesta:** [0.1, 0.9]
> Justifique
Como vimos antes, a largo plazo conviene la estrategia del 50 | 50, por un tema de probabilidades y disminuir el riesgo de perder todos los recursos.

**Respuesta de referencia:** [1, 0]
> Aunque la esperanza aritmética es positiva, lo que importa individualmente es la tasa de crecimiento geométrica: (1.5)^0.5 * (0.6)^0.5 = sqrt(1.5 * 0.6) = sqrt(0.9) ≈ 0.9487 < 1.
La riqueza típica decrece ~5.1% por jugada.

### Teoría de Utilidad Esperada

- Divergencia KL: 4.322

**Tu respuesta:** [0.8, 0.15, 0.05]
> Justifique
Como ya vimos y lo explica la teoria de la utilidad esperada; si la esperanza de los recursos es positiva, si conviene apostar el 50 | 50.

**Respuesta de referencia:** [0, 0, 1]
> A pesar de que la esperanza de los recursos es positiva, la tasa de crecimiento de la riqueza a largo plazo es negativa.
Esto se puede ver en dos pasos, con un Sello y una Cara.
Cuando sale Sello pasamos de 100 a 60, y luego cuando sale Cara pasamos de 60 a 90.
Y como la moneda es normal, a largo plazo vamos a tener la misma cantidad de Caras y Sellos, y la riqueza decrece siempre.

### Fondo común

- Divergencia KL: 0.152

**Tu respuesta:** [0.05, 0.05, 0.9]
> Justifique
Si, conviene. Ya que al aumentar la cantidad de personas, disminuis la incertidumbre, por lo tante disminuye el riesgo.
Como la esperanza es positiva, a largo plazo se espera ganar.

**Respuesta de referencia:** [0, 0, 1]
> Al poner los recursos en un fondo común y dividirlo en partes iguales, la tasa de crecimiento aumenta para todos los participantes.
A medida que el grupo es más grande, la tasa de crecimiento se parece más a la media aritmética, que es positiva, por lo que Sí conviene participar.

### Impuestos

- Divergencia KL: 3.322

**Tu respuesta:** [0.1, 0.1, 0.8]
> Justifique
Aumenta porque al dejar de aportar al fondo común seguimos recibiendo los beneficios del crecimiento colectivo, pero nuestro aporte deja de reducir nuestro capital disponible.
Al tener menos Capital arriesgado y se mantienen las ganancias, esto hace que la tasa de crecimiento efectiva de nuestros recursos sea mayor.

**Respuesta de referencia:** [1, 0, 0]
> Cuando dejamos de aportar al fondo común se reduce la tasa de crecimiento del grupo del cual dependemos y con ella cae nuestra tasa de crecimiento.
Esto se observa matemática y numéricamente.

## Resumen

- **Divergencia KL global:** 29.94
- **Preguntas con divergencia KL infinita** (asignaste 0 a una opción con probabilidad de referencia positiva): 0 de 21
- **Preguntas sin distribución de creencias válida** (reemplazadas por máxima incertidumbre): 0 de 21