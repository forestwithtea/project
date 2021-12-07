# review_NaverMovie_CallbackEmbedding

import pandas as pd
import re
import numpy as np
import tensorflow as tf


def get_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', index_col=0)
    df = df.dropna()
    # df.info()

    x, y = [], []
    for row in zip(df.document.values, df.label.values):
        x.append(clean_str(row[0]))
        y.append(row[1])

    return np.array(x[:10000]), np.array(y[:10000])


# https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # string = re.sub(r",", " ", string)        # custom
    # string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip() if TREC else string.strip().lower()


class TrackEmbedding(tf.keras.callbacks.Callback):
    def __init__(self, layer):
        self.layer = layer

    def on_epoch_end(self, epoch, logs=None):
        print(epoch,
              self.layer.weights[0].shape, self.layer.embeddings.shape,
              self.layer.weights[0][0, 0], self.layer.embeddings[0, 0])


def review_model_embedding():
    x_train, y_train = get_data('data/naver_ratings_train.txt')
    x_test, y_test = get_data('data/naver_ratings_test.txt')

    y_train = np.reshape(y_train, newshape=[-1, 1])
    y_test = np.reshape(y_test, newshape=[-1, 1])

    vocab_size = 2000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_sequence = tokenizer.texts_to_sequences(x_train)
    x_test_sequence = tokenizer.texts_to_sequences(x_test)

    seq_length = 20
    x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequence, maxlen=seq_length)
    x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequence, maxlen=seq_length)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[seq_length]),
        tf.keras.layers.Embedding(vocab_size, 100),
        tf.keras.layers.LSTM(30),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    tracker = TrackEmbedding(model.get_layer(name='embedding'))
    model.fit(x_train_pad, y_train, epochs=10, batch_size=128, verbose=0, callbacks=tracker)


review_model_embedding()
