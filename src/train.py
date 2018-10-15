import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "models/deeplabv3/"))
from model import Deeplabv3
from utils import load_dataset, IMAGE_SIZE
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

FREEZE_LAYER = False
BATCH_SIZE = 4
WEIGHTS = "pascal_voc"
MODEL = "pretrained-noFreeze-size200-SGD"
MODEL_DIR = os.path.join(DIRNAME, "models/deeplabv3/results/{}".format(MODEL))
os.makedirs(MODEL_DIR, exist_ok=True)

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset()

deeplab_model = Deeplabv3(weights=WEIGHTS, input_shape=(200, 200, 3),
                          classes=2)

if FREEZE_LAYER:
    for layer in deeplab_model.layers[:147]:
        layer.trainable = False
# for index, layer in enumerate(deeplab_model.layers):
#     print(index, layer, layer.trainable)

optimizer = SGD(momentum=0.9, clipnorm=1.)

deeplab_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=MODEL_DIR,
                          batch_size=BATCH_SIZE, update_freq="batch")
saver = ModelCheckpoint("{}/model.hdf5".format(MODEL_DIR), verbose=1,
                        save_best_only=True)
stopper = EarlyStopping(patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=0.001)

deeplab_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, verbose=2,
                  validation_data=(X_dev, Y_dev),
                  epochs=100,
                  callbacks=[tensorboard, saver, stopper, reduce_lr])
