# Analisis de resultados

## Evidencia y mejor configuracion

Se compararon cinco corridas del experimento
`bank-marketing-logistic-comparison` sobre la misma particion estratificada
(`seed=42`, 80 % entrenamiento y 20 % prueba). Debido al desbalance de la
variable objetivo (11.7 % de positivos), se definio **average precision (AP)**
como metrica principal antes de comparar los modelos. Los resultados
registrados en MLflow fueron:

| Corrida | C | class_weight | AP | ROC AUC | Exactitud balanceada | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.00 | balanced | 0.4095 | 0.7722 | 0.6977 | 0.2663 | 0.6229 | 0.3731 |
| regularizacion-fuerte | 0.01 | balanced | 0.4076 | 0.7734 | 0.7033 | 0.2581 | 0.6569 | 0.3706 |
| regularizacion-media | 0.10 | balanced | 0.4091 | 0.7726 | 0.7019 | 0.2677 | 0.6333 | 0.3763 |
| regularizacion-debil | 10.00 | balanced | 0.4094 | 0.7722 | 0.6983 | 0.2673 | 0.6229 | 0.3741 |
| sin-balanceo | 1.00 | none | **0.4110** | 0.7719 | 0.5824 | 0.6608 | 0.1767 | 0.2789 |

Con base en la metrica principal, la mejor configuracion fue
**`sin-balanceo` (`C=1`, `class_weight=None`, `max_iter=2000`)**, con AP de
0.4110. Supero al baseline por 0.0016 (0.4 % relativo). Esta diferencia es
pequena y, al provenir de una sola particion, no demuestra por si sola una
mejora generalizable. Ademas, no existe una configuracion dominante en todas
las metricas: `sin-balanceo` ordena ligeramente mejor los casos y alcanza una
precision alta, pero solo recupera 17.7 % de los positivos. Si el objetivo de
negocio fuera detectar mas clientes potenciales, el baseline balanceado seria
mas util por su recall de 62.3 %, aunque no sea el ganador por AP.

## Hiperparametro mas influyente

El hiperparametro con mayor efecto sobre el **comportamiento operativo al
umbral 0.5** fue `class_weight`. La comparacion controlada entre `baseline` y
`sin-balanceo` mantiene iguales `C`, `max_iter`, semilla, datos y umbral. Al
quitar el balanceo, la precision aumento de 0.2663 a 0.6608, pero el recall
cayo de 0.6229 a 0.1767; tambien disminuyeron la exactitud balanceada en 0.1154
y F1 en 0.0942. En cambio, al variar solamente `C` de 0.01 a 10 entre las
corridas balanceadas, AP cambio como maximo 0.0019, la exactitud balanceada
0.0056 y el recall 0.0340. Por tanto, `C` produjo ajustes marginales, mientras
que `class_weight` modifico sustancialmente el tipo de errores del modelo.

Esta conclusion se refiere a las metricas con umbral fijo. Para AP, que no
depende del umbral, ambos hiperparametros tuvieron efectos pequenos y no hay
evidencia suficiente para afirmar una diferencia robusta entre ellos.

## Limitaciones

El ejercicio usa una sola division entrenamiento-prueba y la misma prueba para
comparar configuraciones, lo que puede sobreestimar el desempeno seleccionado;
faltan validacion cruzada, un conjunto final independiente e intervalos de
confianza. Solo se evaluaron cinco configuraciones: `C` y `class_weight` se
variaron de forma aislada, mientras `max_iter=2000` y el umbral 0.5 permanecen
fijos, por lo que no se midieron sus efectos ni interacciones. Tampoco se
incorporaron costos de contacto y conversion para elegir el umbral.

Los datos son historicos y no se realizo validacion temporal ni analisis de
deriva. La regresion logistica limita las relaciones a una frontera lineal
despues de la codificacion, y las categorias `unknown` pueden representar
mecanismos de datos faltantes no estudiados. Finalmente, `duration` se excluyo
correctamente porque no esta disponible antes de terminar la llamada; esto
evita fuga de informacion, pero impide comparar directamente estos resultados
con trabajos que si usan esa variable.
