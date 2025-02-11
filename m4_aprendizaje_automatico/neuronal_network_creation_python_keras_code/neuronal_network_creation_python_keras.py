# Tutorial de Keras, primera red neuronal
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carga del dataset
URL_FICHERO = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

dataset = loadtxt(URL_FICHERO, delimiter=',')

# Separación de las variables independientes de la dependiente
X = dataset[:,0:8]
y = dataset[:, 8]

# Definició del modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilación del modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Ajuste del modelo a los datos
model.fit(X, y, epochs=150, batch_size=10)

# Evaluación del model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' %(accuracy*100))

# Realización de predicciones
predictions = (model.predict(X) > 0.5).astype("int32")

# Obtención de los cinco primeros casos
for i in range(5):
    print('X => %s Ypred => %d (Yreal %d)' % (X[i].tolist(), predictions[i], y[i]))