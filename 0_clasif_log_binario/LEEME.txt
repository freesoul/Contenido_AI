No he tenido mucho tiempo, pero como os prometí, os voy dejando ejemplos mínimos conforme vaya haciéndolos.

El ejemplo que adjunto consiste en la inferencia del sexo (0=mujer,1=hombre) a partir de la edad, el peso, estatura y medidas biométricas de personas aleatorias.

Como podréis comprobar, el código es pequeño y la mayor parte es para preparar los datos, y la parte del modelo es ridícula (literalmente 3 lineas).

Podréis comprobar una eficacia entorno al 75-80% del clasificador logístico binario con las features (variables independientes) que están puestas, pero si le añadís la feature de la medida de la cadera (cometada al lado de la lista) subirá a un 97% o así. Esto se debe a que simplemente hay una relación lineal muy estrecha entre una y otra. 

Hay algunas características que si las añadís empeorarán el resultado. ¿Por qué? Porque al ser un clasificador lineal, no puede lidiar bien con el ruido. Si se tratara de una red neuronal con al menos dos capas y una función de activación no lineal, sí que lidiaría bien con él y no habría efecto.

Hasta la próxima
