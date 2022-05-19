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

url = 'diabetes.csv'
data = pd.read_csv(url)

#Tratamiento de los datos

data.Age.replace(np.nan, 33, inplace=True)
rangos = [20, 35, 50, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.drop(['DiabetesPedigreeFunction', 'BMI', 'Insulin', 'BloodPressure'], axis=1, inplace=True)

# Partir la data por la mitad (Media pa training y media pa testing)

data_train = data[:384]
data_test = data[384:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 

# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

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

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')

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

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

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
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución REGRESIÓN LOGÍSTICA")

precisionLog = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precisionLog}')

recallLog = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallLog}')

f1_scoreLog = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scoreLog}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# MAQUINA DE SOPORTE VECTORIAL CON VALIDACIÓN CRUZADA


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

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionSVC = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionSVC)
plt.title("Mariz de confución MAQUINA DE SOPORTE VECTORIAL")

precisionSVC = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precisionSVC}')

recallSVC = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallSVC}')

f1_scoreSVC = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scoreSVC}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# ARBOL DE DESICION CON VALIDACIÓN CRUZADA


arbol = DecisionTreeClassifier()

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)

print('*'*50)
print('ARBOL DE DESICION Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionarbol = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionarbol)
plt.title("Mariz de confución ARBOL DE DESICION")

precisionarbol = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precisionarbol}')

recallarbol = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallarbol}')

f1_scorearbol = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorearbol}')

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# RANDOM FOREST CON VALIDACIÓN CRUZADA


rf = RandomForestClassifier()

for train, test in kfold.split(x, y):
    rf.fit(x[train], y[train])
    scores_train_train = rf.score(x[train], y[train])
    scores_test_train = rf.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)

print('*'*50)
print('Random Forest Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionrf = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionrf)
plt.title("Mariz de confución RANDOM FOREST")

precisionrf = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precisionrf}')

recallrf = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallrf}')

f1_scorerf = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorerf}')


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# MAQUINA DE SOPORTE VECTORIAL CON VALIDACIÓN CRUZADA


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

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusionkn = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusionkn)
plt.title("Mariz de confución K-Nearest neighbors")

precisionkn = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precisionkn}')

recallkn = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recallkn}')

f1_scorekn = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_scorekn}')
#------------------------------------------------------------------------------------------------