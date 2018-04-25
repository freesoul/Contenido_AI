

import numpy as np
from sklearn import linear_model

###########################
# Leemos los datos
###########################

f = open("dims/bdims.csv")
feature_labels = f.readline().rstrip('\n').split(',')

# cargamos las columnas que queremos
feature_labels_x = ['age', 'wgt', 'hgt', 'wri.di'] # 'hip.gi'
feature_label_y = 'sex'

dataset = np.loadtxt(
            f,
            delimiter=',',
            usecols=[feature_labels.index(c) for c in feature_labels_x + [feature_label_y]]
            )

# mezclamos los datos
np.random.shuffle(dataset)

# separamos el dataset en x e y
data_x = dataset[:, :-1]
data_y = dataset[:, -1:].reshape(-1)

# separamos train y test sets
num_test = int(0.25 * len(dataset))

data_x_train = data_x[:-num_test]
data_y_train = data_y[:-num_test]

data_x_test = data_x[-num_test:]
data_y_test = data_y[-num_test:]



######################################################
# Estamos listos para definir y entrenar el modelo!
######################################################

logreg = linear_model.LogisticRegression()
logreg.fit(data_x_train, data_y_train)

# Ahora comprobamos la eficacia, y nada más!!!
predicciones = logreg.predict(data_x_test)
print("Eficacia del {0}".format(np.mean(predicciones==data_y_test) * 100))
