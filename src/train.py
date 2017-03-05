
# http://aidiary.hatenablog.com/entry/20170110/1484057655
# http://qiita.com/supersaiakujin/items/b9c9da9497c2163d5a74

import argparse
import numpy as np

from keras import backend as k
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from utils.datagen import generator

k.set_image_dim_ordering('tf')

FLAGS = None


def train():
        train_generator, validation_generator = generator()

        # モデルを構築
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # nb_class ... length of classifications: training dataset
        model.add(Dense(train_generator.nb_class, activation='softmax'))

        model.summary()

        # https://keras.io/ja/optimizers/
        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # 訓練
        history = model.fit_generator(
                train_generator,
                samples_per_epoch=2000,
                nb_epoch=FLAGS.nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=800)

        # 成果物
        model.save('model.h5')
        del model


def predict():
        model = load_model('model.h5')

        model.summary()

        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
                'data/test',
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        predictions = model.predict_generator(generator, generator.nb_sample)

        print(predictions)
        print(np.argmax(predictions, axis=1))
        print(predictions.round())

        # out = model.predict(im)
        # best_guess = labels[np.argmax(out)]
        # print(model.predict(training_data).round())


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--nb_epoch', type=int, default=50,
                            help='***')
        FLAGS, unparsed = parser.parse_known_args()
        train()
        predict()
