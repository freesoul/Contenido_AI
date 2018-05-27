
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

##############################
# Cargamos los datos
##############################
iris = load_iris()
headers = iris.target_names
perm = np.random.permutation(len(iris.data)) # para mezclar
data_x = iris.data[perm]
data_y = iris.target[perm]

#######################################
# Seleccionamos las mejores features.
# A la hora de usarlo de verdad, hay
# que entrenar estos procesadores solo
# con el train set.
#######################################

# Que haya varianza (sino no representan buen cambio)
data_x = VarianceThreshold(threshold=0.35).fit_transform(data_x)

# Que estén correlacionados con los datos destino
data_x = SelectKBest(chi2,k=2).fit_transform(data_x, data_y)

# Reducimos variables independientes redundantes o poco importantes
pca = decomposition.PCA(n_components=1)
pca.fit(data_x)
data_x = pca.transform(data_x)


#######################################
# Separamos train y test sets
#######################################
num_test = int(0.25 * len(data_x))

data_x_train = data_x[:-num_test]
data_y_train = data_y[:-num_test]

data_x_test = data_x[-num_test:]
data_y_test = data_y[-num_test:]


#######################################
# Modelo de regresión
#######################################


logreg = LogisticRegression(multi_class='ovr')
logreg.fit(data_x_train, data_y_train)

# Comprobamos
predictions = logreg.predict(data_x_test)
print(data_y_test)
print(predictions)
print("Log Accuracy: {0}%".format(np.mean(predictions==data_y_test)* 100))


#######################################
# MLP
#######################################

# No muestra mayor eficacia que logreg porque el problema es linealmente separable.
# A veces muestra peor eficacia, porque se queda en mínimos locales.


MLP = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(4,),
                    max_iter=1e3)
MLP.fit(data_x_train, data_y_train)

# Comprobamos
predictions = MLP.predict(data_x_test)
print(data_y_test)
print(predictions)
print("MLP Accuracy: {0}%".format(np.mean(predictions==data_y_test)* 100))
