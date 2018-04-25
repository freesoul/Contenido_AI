
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier

##############################
# Cargamos los datos
##############################

f = open("mushrooms.csv")
headers = f.readline().strip('\n').split(',')[1:]
data = np.loadtxt(f, dtype='str', delimiter=',')
np.random.shuffle(data)
data_x = data[:, 1:]
data_y = data[:, 0].ravel() # Ravel, al igual que reshape(-1), nos pasa de columna a vector

#######################################
# Codificamos las variables categóricas
#######################################
encoders = [LabelBinarizer() for i in range(len(headers))] # Uno por feature

data_features = data_x.transpose()
data_x_encoded = []
for i in range(len(headers)):
    data_x_encoded.append(encoders[i].fit_transform(data_features[i]))

data_x_encoded = np.concatenate(data_x_encoded, axis=1) # Unimos las features


#######################################
# Seleccionamos las mejores features.
# A la hora de usarlo de verdad, hay
# que entrenar estos procesadores solo
# con el train set.
#######################################

# Que haya varianza (sino no representan buen cambio)
data_x_encoded = VarianceThreshold(threshold=0.15).fit_transform(data_x_encoded)

# Que estén correlacionados con los datos destino
data_x_encoded = SelectKBest(chi2,k=10).fit_transform(data_x_encoded, data_y)

# Reducimos variables independientes redundantes o poco importantes
pca = decomposition.PCA(n_components=3)
pca.fit(data_x_encoded)
data_x_encoded = pca.transform(data_x_encoded)


#######################################
# Separamos train y test sets
#######################################
num_test = int(0.25 * len(data_x_encoded))

data_x_train = data_x_encoded[:-num_test]
data_y_train = data_y[:-num_test]

data_x_test = data_x_encoded[-num_test:]
data_y_test = data_y[-num_test:]


#######################################
# Modelos
#######################################

# Modelo de regresión
logreg = LogisticRegression()
logreg.fit(data_x_train, data_y_train)

# Comprobamos
predictions = logreg.predict(data_x_test)
print("Log Accuracy: {0}%".format(np.mean(predictions==data_y_test)* 100))


# # MLP
# # No muestra mayor eficacia que logreg porque el problema es linealmente separable.
# # A veces muestra peor eficacia, porque se queda en mínimos locales.
#
# MLP = MLPClassifier(hidden_layer_sizes=int(len(data_x_test[0])/2))
# MLP.fit(data_x_train, data_y_train)
#
# # Comprobamos
# predictions = MLP.predict(data_x_test)
# print("MLP Accuracy: {0}%".format(np.mean(predictions==data_y_test)* 100))
