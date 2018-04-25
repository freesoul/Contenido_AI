
A diferencia de en el último ejemplo, en este no empleamos variables independientes continuas, sino que usamos categorías.

Además, en este ejemplo hacemos un paso extra: seleccionamos los datos.

En términos normales, la forma de seleccionar los datos se entrena únicamente en una fracción de los datos, y luego se comprueba
en la fracción reservada para testeo (para prevenir overfitting al testset). El overfitting es sobreajuste a un conjunto de datos.

No obstante, en este ejemplo, por simplificación, seleccionamos los datos previamente a la separación en train y test set.

Las formas en que seleccionamos los datos son tres:

- Reducción de dimensionalidad y eliminación de datos correlacionados (PCA).
- Eliminación de datos sin buena varianza (si los datos son siempre iguales, no ayudan a clasificar nada; VarianceThreshold)
- Seleccionar los datos mejor correlacionados a las clases destino. (SelectKBest)


Hasta pronto
JFK


Dataset: champiñones venenosos (Kaggle) https://www.kaggle.com/uciml/mushroom-classification
