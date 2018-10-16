from train import train
from utils import load_dataset


imageSizes = [100, 200, 500]

for size in imageSizes:
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
