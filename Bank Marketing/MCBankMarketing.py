import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #libreria del modelo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score




simplefilter(action='ignore', category=FutureWarning)

url = 'bank-full.csv'
data = pd.read_csv(url)

#Tratamiento de los datos

data.marital.replace(['married', 'single', 'divorced'], [2, 1, 0], inplace= True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace= True)
data.default.replace(['no', 'yes'], [0, 1], inplace= True)
data.housing.replace(['no', 'yes'], [0, 1], inplace= True)
data.loan.replace(['no', 'yes'], [0, 1], inplace= True)
data.y.replace(['no', 'yes'], [0, 1], inplace= True)
data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace= True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace= True)

data.drop(['balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'job'], axis=1, inplace=True)
data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0, how='any', inplace=True)

# Partir la data en dos

data_train = data[:22605]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) # 0 desconocido 1 fallo 2 otro 3 exito

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_train.drop(['y'], 1))
y_test_out = np.array(data_train.y) # 0 desconocido 1 fallo 2 otro 3 exito

#------------------------------------------------------------------------------------------------
# Regresi??n Log??stica

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# M??TRICAS

print('*'*50)
print('Regresi??n Log??stica')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {logreg.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------

# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# M??TRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {svc.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------
# ARBOL DE DECISI??N

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# M??TRICAS

print('*'*50)
print('Decisi??n Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {arbol.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------
# RANDOM FOREST

rf = RandomForestClassifier()

# Entrenar el modelo
rf.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')


# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {rf.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------
# K-Nearest neighbors

# Seleccionar un modelo

kn = KNeighborsClassifier()

# Entrenar el modelo

kn.fit(x_train, y_train)

# Metricas

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {kn.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {kn.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {kn.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# REGRESI??N LOG??STICA CON VALIDACI??N CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresi??n Log??stica Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confuci??n REGRESI??N LOG??STICA")

precisionLog = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precisionLog}')

recallLog = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallLog}')

f1_scoreLog = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scoreLog}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# MAQUINA DE SOPORTE VECTORIAL CON VALIDACI??N CRUZADA


svc = SVC(gamma='auto')

for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('MAQUINA DE SOPORTE VECTORIAL cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionSVC = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionSVC)
plt.title("Mariz de confuci??n MAQUINA DE SOPORTE VECTORIAL")

precisionSVC = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precisionSVC}')

recallSVC = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallSVC}')

f1_scoreSVC = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scoreSVC}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# ARBOL DE DESICION CON VALIDACI??N CRUZADA


arbol = DecisionTreeClassifier()

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)

print('*'*50)
print('ARBOL DE DESICION Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionarbol = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionarbol)
plt.title("Mariz de confuci??n ARBOL DE DESICION")

precisionarbol = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precisionarbol}')

recallarbol = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallarbol}')

f1_scorearbol = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorearbol}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# RANDOM FOREST CON VALIDACI??N CRUZADA


rf = RandomForestClassifier()

for train, test in kfold.split(x, y):
    rf.fit(x[train], y[train])
    scores_train_train = rf.score(x[train], y[train])
    scores_test_train = rf.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)

print('*'*50)
print('Random Forest Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {rf.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionrf = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionrf)
plt.title("Mariz de confuci??n RANDOM FOREST")

precisionrf = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precisionrf}')

recallrf = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallrf}')

f1_scorerf = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorerf}')


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# MAQUINA DE SOPORTE VECTORIAL CON VALIDACI??N CRUZADA


kn = KNeighborsClassifier()

for train, test in kfold.split(x, y):
    kn.fit(x[train], y[train])
    scores_train_train = kn.score(x[train], y[train])
    scores_test_train = kn.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = kn.predict(x_test_out)

print('*'*50)
print('K-Nearest neighbors validacion cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {kn.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionkn = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionkn)
plt.title("Mariz de confuci??n K-Nearest neighbors")

precisionkn = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precisionkn}')

recallkn = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallkn}')

f1_scorekn = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorekn}')


#------------------------------------------------------------------------------------------------

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')

#Final uwu