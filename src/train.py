import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "models/deeplabv3/"))
from model import Deeplabv3
from utils import jacard_coef
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import backend as K


def train(X_train, Y_train, X_dev, Y_dev, batch_size=4,
          freeze=False, pretrained=True, model="test",
          aug=False, generator=None, n_epochs=100):

    model_dir = os.path.join(DIRNAME, "models/deeplabv3/results/{}".format(model))
    os.makedirs(model_dir, exist_ok=True)

    if pretrained:
        weights = "pascal_voc"
    else:
        weights = None

    deeplab_model = Deeplabv3(weights=weights, input_shape=(X_train.shape[1:]),
                              classes=2)

    if freeze:
        for layer in deeplab_model.layers[:147]:
            layer.trainable = False

    optimizer = SGD(momentum=0.9, clipnorm=1.)

    deeplab_model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy', jacard_coef])

    tensorboard = TensorBoard(log_dir=model_dir,
                              batch_size=batch_size, update_freq="batch")
    saver = ModelCheckpoint("{}/model.hdf5".format(model_dir), verbose=1,
                            save_best_only=True, monitor="val_jacard_coef",
                            mode="max")
    stopper = EarlyStopping(patience=50, verbose=1, monitor="val_acc",
                            mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5,
                                  patience=5, verbose=1, min_lr=0.001)

    if aug:
        if generator is None:
            print("Necessário fornecer gerador de image augmentation")
            return

        deeplab_model.fit_generator(generator, verbose=2,
                                    validation_data=(X_dev, Y_dev),
                                    epochs=n_epochs,
                                    callbacks=[tensorboard, saver, stopper, reduce_lr],
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    validation_steps=X_dev.shape[0] // batch_size)
    else:
        deeplab_model.fit(X_train, Y_train, batch_size=batch_size, verbose=2,
                          validation_data=(X_dev, Y_dev),
                          epochs=n_epochs,
                          callbacks=[tensorboard, saver, stopper, reduce_lr])
    print("Modelo {} treinado!".format(model))
