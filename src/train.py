
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

        model = Sequential()

        # conv1
        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # conv2
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # one dim
        model.add(Flatten())

        # fc1
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # fc2
        # dim ... length of classifications: training dataset
        model.add(Dense(train_generator.nb_class, activation='softmax'))

        # model visualize
        model.summary()

        # https://keras.io/ja/optimizers/
        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # learning
        history = model.fit_generator(
                train_generator,
                samples_per_epoch=500,
                nb_epoch=FLAGS.nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=200)

        # model store
        model.save('model.h5')
        del model


def predict():
        # label list: training dataset
        train_generator, validation_generator = generator()
        labels = dict((v, k) for k, v in train_generator.class_indices.items())

        # model load
        model = load_model('model.h5')
        model.summary()

        # test data: don't shuffle
        datagen = ImageDataGenerator(rescale=1./255)
        test_generator = datagen.flow_from_directory(
                'data/test',
                shuffle=False,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        # predict
        predictions = model.predict_generator(test_generator, test_generator.nb_sample)

        # best label index
        y = np.argmax(predictions, axis=1)

        # result
        for (file, index) in zip (test_generator.filenames, y):
                print('result: "%s" predict "%s" class.' % (file, labels[index]))


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--nb_epoch', type=int, default=50,
                            help='***')
        FLAGS, unparsed = parser.parse_known_args()
        train()
        predict()
