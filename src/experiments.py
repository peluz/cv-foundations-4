from train import train
from utils import load_dataset


imageSizes = [100, 200, 250, 500]

for size in imageSizes:
    if size == 100:
        small_batch = 8
        big_batch = 32
    else:
        small_batch = 4
        big_batch = 16

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset(size)
    train(X_train, Y_train, X_dev, Y_dev, batch_size=4,
          freeze=False, pretrained=True,
          model="pretrained-noFreeze-size{}-SGD".format(size))
    train(X_train, Y_train, X_dev, Y_dev, batch_size=16,
          freeze=True, pretrained=True,
          model="pretrained-Freeze-size{}-SGD".format(size))
    train(X_train, Y_train, X_dev, Y_dev, batch_size=4,
          freeze=False, pretrained=False,
          model="noPretrained-noFreeze-size{}-SGD".format(size))
