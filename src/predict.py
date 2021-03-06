import argparse
import numpy as np
import utils.datagen as datagen

from keras import backend as k
from keras.models import load_model

k.set_image_data_format('channels_last')

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
    predictions = model.predict_generator(test_generator, test_generator.samples)

    # result
    for (file, prediction) in zip(test_generator.filenames, predictions):
        print('')
        print('##############################')
        print('')
        print('result: "%s" predict ...' % (file))
        ys = np.argsort(prediction)[::-1][:5]
        for y in ys:
            print('> %f "%s" class.' % (prediction[y], labels[y]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='***')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='***')
    FLAGS, unparsed = parser.parse_known_args()
    predict()
