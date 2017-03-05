import argparse
import numpy as np
import utils.datagen as datagen

from keras import backend as k
from keras.models import load_model

k.set_image_dim_ordering('tf')

FLAGS = None


def predict():
        # label list: training dataset
        train_generator, validation_generator = datagen.train()
        labels = dict((v, k) for k, v in train_generator.class_indices.items())

        # model load
        model = load_model(FLAGS.model)
        model.summary()

        # test data
        test_generator = datagen.test()

        # predict
        predictions = model.predict_generator(test_generator, test_generator.nb_sample)

        # best label index
        y = np.argmax(predictions, axis=1)

        # result
        for (file, index) in zip(test_generator.filenames, y):
                print('result: "%s" predict "%s" class.' % (file, labels[index]))


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='model.h5',
                            help='***')
        FLAGS, unparsed = parser.parse_known_args()
        predict()
