# Testado no Tensorflow 2.18

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import keras
from keras import layers

""" 
Ler um ficheiro com o dataset wineData - este data set contém 13 atributos
(features) referentes a vinhos italianos de três produtores diferentes.
A ideia é identificar o produtor do vinho com base nos atributos do vinho.
Os atributos são os seguintes (https://archive.ics.uci.edu/dataset/109/wine):

 1) Alcohol
 2) Malic acid
 3) Ash
 4) Alcalinity of ash
 5) Magnesium
 6) Total phenols
 7) Flavanoids
 8) Nonflavanoid phenols
 9) Proanthocyanins
 10) Color intensity
 11) Hue
 12) OD280/OD315 of diluted wines
 13) Proline """

##################################################################
# Ler e preparar o dataset para ser compatível com o tensorflow

# leitura do ficheiro com os dados
data = np.array(pd.read_csv("wineData.csv", sep=";"))

# n. de amostras e n. de features
nSamples = data.shape[0]
nFeatures = data.shape[1]-1

# "baralhar" as amostras (mas sempre da mesma maneira)
SEED = 12345
np.random.seed(SEED)
data = np.random.permutation(data)

# separar as features das classificações (labels)
features = data[:, 0:nFeatures]
labels = data[:, nFeatures].astype(int) - 1

# colocar as labels no formato adequado para treino (matriz N x C)
labels = keras.utils.to_categorical(labels, 3)

# divisão treino/validação (80% treino - 20% validação)
SPLIT = features.shape[0] * 8 // 10
x_train = features[:SPLIT, :]
y_train = labels[:SPLIT, :]
x_val = features[SPLIT + 1:, :]
y_val = labels[SPLIT + 1:, :]


##################################################################
# Definição, compilação e treino do modelo

# Inicialização dos pesos da rede - na prática este passo não se costuma fazer.
# Incluiu-se apenas para garantir que os pesos sao sempre inicializados da mesma maneira
# a fim de se poder comparar melhor o efeito das alterações aos hiper-parâmetros pedidas
# num dos exercícios
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED)

# definição da arquitetura da rede neuronal
wine_model = keras.Sequential([
    layers.Input((13,)),
    layers.Dense(20, activation='relu', kernel_initializer=initializer),
    layers.Dense(3, activation='softmax', kernel_initializer=initializer)
])

# mostrar um sumário do modelo (organização e n. de pesos a otimizar em cada camada)
wine_model.summary()

# compilar o modelo, definindo a loss function e o algoritmo de otimização
wine_model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

# treinar, guardando os dados do treino numa variável
h = wine_model.fit(x_train, y_train, batch_size=16, epochs=500,
                        validation_data=(x_val, y_val))

##################################################################
# Obter e mostrar resultados

# obter os id's das classes verdadeiras
y_true = np.argmax(y_val, axis=1)

# realizar as predições e obter os id's das classes preditas
output_pred = wine_model.predict(x_val)    # ou então, output_pred = wineModel(x_val)
y_pred = np.argmax(output_pred, axis=1)

# gerar uma matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# mostrar figuras - accuracy, loss e matriz de confusão
plt.figure(num=1)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.figure(num=2)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylim(0, 2.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Produtor A", "Produtor B", "Produtor C"])
disp.plot(cmap=plt.colormaps['Oranges'])
plt.show()
