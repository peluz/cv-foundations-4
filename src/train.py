import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "models/keras-deeplab-v3-plus-master/"))
from model import Deeplabv3
from utils import load_dataset

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset()

deeplab_model = Deeplabv3(input_shape=(200, 200, 3), classes=2)
for layer in deeplab_model.layers[:147]:
    layer.trainable = False
for index, layer in enumerate(deeplab_model.layers):
    print(index, layer, layer.trainable)

deeplab_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

deeplab_model.fit(X_train, Y_train, batch_size=16, verbose=2,
                  validation_data=(X_dev, Y_dev))
