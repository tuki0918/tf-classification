from keras.preprocessing.image import ImageDataGenerator


def generator():
    # 訓練データとバリデーションデータを生成するジェネレータを作成
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        # color_mode='grayscale',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        # color_mode='grayscale',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    return train_generator, validation_generator
