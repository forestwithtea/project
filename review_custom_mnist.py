# review_custom_mnist

import tensorflow as tf


def model_custom_mnist(target_size):
    gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    get_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    flow_train = gen_train.flow_from_directory('data/custom_mnist/train',
                                               target_size=target_size,
                                               batch_size=32,
                                               class_mode='sparse')
    flow_test = get_test.flow_from_directory('data/custom_mnist/test',
                                             target_size=target_size,
                                             batch_size=32,
                                             class_mode='sparse')

    conv_base = tf.keras.applications.VGG16(include_top=False, input_shape=[target_size[0], target_size[1], 3])
    conv_base.trainable = False

    model = tf.keras.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(flow_train, epochs=10, verbose=2, validation_data=flow_test)


model_custom_mnist(target_size=[100, 100])
