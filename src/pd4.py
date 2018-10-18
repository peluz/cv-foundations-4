import argparse
from train import train
from utils import load_dataset, load
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(
    description="Image Segmentation")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--r1", action="store_true",
                   help="Requisito 1")
group.add_argument("--r2", action="store_true",
                   help="Requisito 2")
parser.add_argument("--imageSize", choices=(100, 200, 250),
                    help="Tamanho da imagem", type=int, default=200)
parser.add_argument("--batchSize", help="Tamanho do batch", type=int,
                    default=4)
parser.add_argument("--freeze", action="store_true",
                    help="Não treinar camadas do encoder",
                    default=False)
parser.add_argument("--randomInit", help="Não usar pesos pretreinados",
                    action="store_true", default=False)
parser.add_argument("--train", help="Treinar modelo",
                    action="store_true", default=False)
parser.add_argument("--model", help="Nome do modelo a ser salvo/carregado",
                    type=str, default="test")
parser.add_argument("--dataAug", help="Treinar com data agumentation",
                    action="store_true", default=False)


def main(r1, r2, train_model, image_size,
         batch_size, freeze, pretrained, model, aug):
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_dataset(image_size)
    if r1:
        pass
    elif r2:
        if train_model:
            if aug:
                datagen = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True)
                datagen.fit(X_train)
                train_generator = datagen.flow(X_train,
                                               Y_train,
                                               batch_size=batch_size)
                train(X_train, Y_train, X_dev, Y_dev, batch_size=batch_size,
                      freeze=freeze, pretrained=pretrained, model=model, aug=True,
                      generator=train_generator)
            else:
                train(X_train, Y_train, X_dev, Y_dev, batch_size=batch_size,
                      freeze=freeze, pretrained=pretrained, model=model)
        model = load(model)
        results = model.evaluate(X_test, Y_test, batch_size)
        print(results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.r1, args.r2, args.train, args.imageSize, args.batchSize,
         args.freeze, not args.randomInit, args.model, args.dataAug)
