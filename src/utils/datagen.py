from keras.preprocessing.image import ImageDataGenerator


def train(path='data'):
    # train, validation data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '{}/train'.format(path),
        # color_mode='grayscale',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        '{}/validation'.format(path),
        # color_mode='grayscale',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    return train_generator, validation_generator


def test(path='data'):
    # test data: don't shuffle
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        '{}/test'.format(path),
        shuffle=False,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    return test_generator
