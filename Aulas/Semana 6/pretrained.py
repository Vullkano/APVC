# Testado com tensorflow 2.19, keras 3.9 e Python 3.12

from keras.src.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import utils
import numpy as np

# outras alternativas de import da VGG16:
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# obter o modelo VGG16, pre-treinado com o dataset ImageNet
vgg16Model = VGG16(weights='imagenet', classes=1000)

# constantes - 224x224 e' a dimensao das imagens de input na rede VGG16
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_FILE = './sample_images/banana1.jpg'

# obter uma imagem de teste - mudar a gosto - devolve uma imagem em formato PIL
img = utils.load_img(IMG_FILE, target_size=(IMG_HEIGHT, IMG_WIDTH))

# pre-processamento
x = utils.img_to_array(img)       # converte a imagem de PIL para numpy array
x = np.expand_dims(x, axis=0)     # cria uma dimensao adicional, apenas para para ser compativel com o formato de entrada na CNN
x = preprocess_input(x)           # usa o pre-processamento proprio da VGG16 (converte as imagens para BGR
                                  # e subtrai-lhes um valor pré-definido)
# obter as predicoes
preds = vgg16Model.predict(x)

# processa as predicoes de forma a obter as 5 classes mais provaveis
decoded_preds = decode_predictions(preds, top=5)[0]

# mostra os resultados da classificacao
print('Top-5 Class predictions: ')
for class_name, class_description, score in decoded_preds:
    print(class_description, ": ", score)