from train import train
from utils import load_dataset
from keras.preprocessing.image import ImageDataGenerator

size = 250
batch = 4

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset(size)

datagen.fit(X_train)

train_generator = datagen.flow(X_train,
                               Y_train,
                               batch_size=batch)

train(X_train, Y_train, X_dev, Y_dev, batch_size=batch,
      freeze=False, pretrained=True,
      model="pretrained-noFreeze-size{}-SGD-Aug".format(size),
      aug=True, generator=train_generator,
      n_epochs=1000)
