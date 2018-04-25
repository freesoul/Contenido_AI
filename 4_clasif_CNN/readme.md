
Dataset empleado: CIFAR-100 (NO CIFAR-10, 100. Primer resultado de google)

Del dataset se han creado 3 supergrupos de interés a partir de la fusión de distintos subgrupos del CIFAR-100. Estos grupos son: árboles, personas y vehiculos.


Se ha definido una red neuronal con Keras, en la cual hay varias capas convolucionales, dos max poolings y dos dropouts.


Cada capa convolucional tiene una ventana que va iterando por la imagen (el input de estas capas es siempre Altura x Anchura x Profundidad). Son redes neuronales de toda la vida solo que se mueven por la imagen, y reconstruyen una nueva. La anchura y la altura de la nueva imagen siempre es menor que la original (Wf = W0 - (Wventana - 1), y lo mismo correspondiente con la altura). La profundidad nueva es el número de neuronas de la capa convolucional, además correspondiendo cada neurona con una feature extraída por la ventana (no hay 1 ventana, sino tantas ventanas que iteran la imagen como neuronas), lo cual también se puede llamar filtro o plantilla (plantilla porque el dot product de la plantilla x la imagen mide la similitud).

Las función de activación utilizada tras las capas convolucionales (que de por si son neuronas o perceptrones sin función de activación, y cuyo output es solo el producto escalar del input por los pesos asociados a cada neurona) es RelU (ver google), una función no lineal (igual que podría ser una sigmoidal, como tanh). 


Los maxpoolings reducen la resolución de la imagen, de forma que un maxpooling con ventana de (2,2) reduce el número de ventanitas a la mitad, y con una ventana de (3,3), a un tercio. Pero de una forma particular: de cada ventana se selecciona el valor más representativo (y ocurre a lo largo de todas las características o features generadas, en toda la profundidad de la imagen de datos).


Los dropouts eliminan datos al azar para evitar overfitting y permitir que el aprendizaje se centre en encontrar las generalidades más que en patrones de imágenes concretas.


La capa Flatten es una capa neuronal normal y corriente solo que toma como input datos que no son un vector sino una matriz.

La capa Dense es nuestra capa final, y como activación le ponemos una función softmax que da probabilidades para cada clase final.


Lo último que hace el código es guardar el modelo, para poder predecir con él tener que volver a entrenar, y dar la eficacia.
La eficacia conseguida con 20 epochs (rondas de aprendizaje) con batches de 100 (número de imágenes enseñadas a la vez al algoritmo de backpropagation) es de 80%. 

Con tuneado de hiperparámetros (búsqueda por ejemplo exhaustiva mediante GridSearch de los parámetros óptimos para el modelo) se puede mejorar mucho seguramente la eficiencia. El tuneado puede incluir el optimizador empleado, el número de neuronas en las capas, el dropout, el tamaño de las ventanas, etc. El tuneado de hiperparámetros debe hacerse en un subset del trainset (para prevenir overfitting de los hiperparámetros al trainset en si o al testset), o bien si no se quiere reducir la disposición de datos para entrenamiento previo al uso del testset, mediante crossvalidation (ver wikipedia).


Hasta pronto



