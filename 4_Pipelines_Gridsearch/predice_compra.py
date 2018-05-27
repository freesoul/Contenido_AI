
import numpy as np

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier



TRAIN_SET = 0.75



##############################
# Cargamos los datos
##############################

f = open("bank/bank-full.csv") # bank.csv

# Obtenemos las columnas
headers = [name.strip("\"") for name in f.readline().strip('\n').split(';')]

# definimos los tipos de datos
categoricas = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'campaign', 'contact', 'poutcome', 'y']
continuas = ['age','balance','day','month','duration','pdays','previous']

# obtenemos indices de las columnas
categoricas_indices = [headers.index(c) for c in categoricas]
continuas_indices = [headers.index(c) for c in continuas]



##############################
# Cargamos los datos
##############################

# Eliminamos las comillas
converters =  dict((i,lambda s: s.strip(b"\"")) for i in range(len(headers)))

# Pasamos 'month' a variable numérica
meses_a_numero = {'apr': 4, 'sep': 9, 'mar': 3, 'may': 5, 'jun': 6, 'aug': 8, 'feb': 2, 'nov': 11, 'jan': 1, 'dec': 12, 'oct': 10, 'jul': 7}
converters[headers.index('month')] = lambda s: meses_a_numero[s.strip(b"\"").decode('utf-8')]

# Cargamos
data = np.loadtxt(
    f,
    dtype='str',
    delimiter=';',
    converters = converters
)

# Mezclamos
np.random.shuffle(data)

# Separamos X de Y

data_x = data[:,:-1]
data_y = data[:,-1]

print("Shape del dataset (x): {0}".format(data_x.shape))
print("Shape del dataset (y): {0}".format(data_y.shape))



#######################################
# Codificamos las variables categóricas
#######################################

encoders = dict((i,LabelBinarizer()) for i in categoricas_indices)

data_features = data_x.transpose()

data_features_new = []
for i in range(len(data_features)):
    if i in encoders.keys():
        data_features_new.append(encoders[i].fit_transform(data_features[i]))
    else:
        data_features_new.append([[x] for x in data_features[i]])

data_x = np.concatenate(data_features_new, axis=1)


print("Shape del dataset codificado (x): {0}\n".format(data_x.shape))

# # # También codificamos la "y"
# data_y_encoder = LabelBinarizer().fit(data_y)
# data_y = data_y_encoder.transform(data_y)



#######################################
# Separamos train y test set
#######################################
num_train_set = int(data_x.shape[0] * TRAIN_SET)

data_x_train = data_x[:num_train_set].astype(np.float64)
data_y_train = data_y[:num_train_set].reshape(-1,1)

data_x_test = data_x[num_train_set:].astype(np.float64)
data_y_test = data_y[num_train_set:].reshape(-1,1)

print("Shape del trainset (x): {0}".format(data_x_train.shape))
print("Shape del trainset (y): {0}".format(data_y_train.shape))
print("Shape del testset (x): {0}".format(data_x_test.shape))
print("Shape del testset (y): {0}\n".format(data_y_test.shape))



#######################################
# Función para medir score
#######################################

def loss_func(y, y_pred):
    indices_yes = np.argwhere(y.ravel()=='yes').ravel()
    predichos_yes = y_pred[indices_yes]
    error = np.sum(predichos_yes=='no') / predichos_yes.shape[0]
    return np.log(1 + error)

scorer = make_scorer(loss_func)


#######################################
# Regresión logística
#######################################


print("="*80)
print("= Regresión logística")
print("="*80)

pipeline = Pipeline([
    # ('filtra_varianza', VarianceThreshold()),
    ('reduce_dim', PCA()),
    # ('normalize', MinMaxScaler(feature_range=(0,1)) ),
    # ('selectkbest', SelectKBest(chi2)),
    ('logreg', LogisticRegression())
])

param_grid = [
    {
        # 'filtra_varianza__threshold': [0.10,0.15,0.20,0.25,0.30],
        'reduce_dim__n_components': [1,5,10,30,60],
        # 'selectkbest__k':[1,3,5,20,40]
    }
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=3, param_grid=param_grid, scoring=scorer)
grid.fit(data_x_train, data_y_train)

for score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
    print("Score {0} para params {1}\n".format(score,params))

testset_score = round(grid.score(data_x_test, data_y_test)*100,2)
print("Eficacia Reg. log. (test set): {0}%\n".format(testset_score))



#######################################
# SVC
#######################################

print("="*80)
print("= SVC")
print("="*80)

pipeline = Pipeline([
    # ('filtra_varianza', VarianceThreshold()),
    # ('reduce_dim', PCA()),
    ('normalize', MinMaxScaler(feature_range=(-1,1)) ),
    # ('selectkbest', SelectKBest(chi2)),
    ('SVC', SVC())
])


param_grid = [
    {
        # 'filtra_varianza__threshold': [0.10,0.15,0.20,0.25,0.30],
        # 'reduce_dim__n_components': [1,5,10,20,30],
        # 'selectkbest__k':[1,5]
        'SVC__C': [0.01, 10], # [0.01,1,10,100]
        'SVC__kernel':['rbf'] # ['linear','rbf']
    }
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=3, param_grid=param_grid,scoring=scorer)
grid.fit(data_x_train, data_y_train)

for score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
    print("Score {0} para params {1}\n".format(score,params))

testset_score = round(grid.score(data_x_test, data_y_test)*100,2)
print("Eficacia SVC (test set): {0}%\n".format(testset_score))


#######################################
# MLP
#######################################

print("="*80)
print("= MLP")
print("="*80)

pipeline = Pipeline([
    # ('filtra_varianza', VarianceThreshold()),
    # ('reduce_dim', PCA()),
    # ('normalize', MinMaxScaler(feature_range=(0,1)) ),
    # ('selectkbest', SelectKBest(chi2)),
    ('MLP', MLPClassifier()) # ,max_iter=1e3 # lbfgs
])


param_grid = [
    {
        # 'filtra_varianza__threshold': [0.10,0.20],
        # 'reduce_dim__n_components': [5,15,30],
        # 'selectkbest__k':K_BEST,
        'MLP__max_iter': [1000], # [1000, 5000, 10000]
        'MLP__solver': ['lbfgs'],#['lbfgs', 'adam'],
        'MLP__hidden_layer_sizes': [(5,)] #  [(5,),(15,),(30,)]
    }
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=3, param_grid=param_grid, scoring=scorer)
grid.fit(data_x_train, data_y_train)

for score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
    print("Score {0} para params {1}\n".format(score,params))

testset_score = round(grid.score(data_x_test, data_y_test)*100,2)
print("Eficacia MLP (test set): {0}%\n".format(testset_score))



#######################################
# Decision tree
#######################################

print("="*80)
print("= Árbol de decisiones")
print("="*80)

pipeline = Pipeline([
    # ('filtra_varianza', VarianceThreshold()),
    # ('reduce_dim', PCA()),
    ('normalize', MinMaxScaler(feature_range=(0,1)) ),
    ('selectkbest', SelectKBest(chi2)),
    ('DecisionTreeClassifier', DecisionTreeClassifier())
])

param_grid = [
    {
        # 'filtra_varianza__threshold': [0.10,0.25],
        # 'reduce_dim__n_components': [1,2,3,4],
        'selectkbest__k':[1,3,5]
    }
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=3, param_grid=param_grid, scoring=scorer)
grid.fit(data_x_train, data_y_train)

features = grid.best_estimator_.named_steps['selectkbest']
print(features.scores_.argmax()) # 66 -> Una de las categorías de 'poutcome' (success)

for score, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
    print("Score {0} para params {1}\n".format(score,params))


testset_score = round(grid.score(data_x_test, data_y_test)*100,2)
print("Eficacia árbol decisiones (test set): {0}%\n".format(testset_score))
