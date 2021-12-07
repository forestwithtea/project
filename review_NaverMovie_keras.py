# review_NaverMovie_keras


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


def review_model_origin():
    x_train, y_train = get_data('data/naver_ratings_train.txt')
    x_test, y_test = get_data('data/naver_ratings_test.txt')

    y_train = np.reshape(y_train, newshape=[-1, 1])
    y_test = np.reshape(y_test, newshape=[-1, 1])

    # print(x_train[:3])
    # print(y_train[:3])

    vocab_size = 2000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_sequence = tokenizer.texts_to_sequences(x_train)
    x_test_sequence = tokenizer.texts_to_sequences(x_test)

    # print(x_train_sequence[:3])
    # print(type(x_train_sequence))

    seq_length = 20
    x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequence, maxlen=seq_length)
    x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequence, maxlen=seq_length)

    # print(x_train_pad[0])
    # [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  39 353   4
    #  810 811]
    # print(x_train_pad.shape, x_test_pad.shape)      # (2000, 20) (2000, 20)

    onehot = np.identity(vocab_size, dtype=np.int32)

    x_train_onehot = onehot[x_train_pad]
    x_test_onehot = onehot[x_test_pad]

    # print(x_train_onehot.shape, x_test_onehot.shape)        # (2000, 20, 2000) (2000, 20, 2000)
    print(x_train_onehot)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[seq_length, vocab_size]))
    model.add(tf.keras.layers.LSTM(30))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    model.fit(x_train_onehot, y_train, epochs=10, batch_size=128, verbose=2)

    print(model.evaluate(x_test_onehot, y_test, verbose=0))


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
    x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequence, maxlen=seq_length)
    x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequence, maxlen=seq_length)

    # print(x_train_pad.shape, x_test_pad.shape)      # (2000, 20) (2000, 20)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[seq_length]),
        tf.keras.layers.Embedding(vocab_size, 100),
        tf.keras.layers.LSTM(30, return_sequences=True),
        tf.keras.layers.LSTM(30),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    model.fit(x_train_pad, y_train, epochs=10, batch_size=128, verbose=2)

    print(model.evaluate(x_test_pad, y_test, verbose=0))
    preds = model.predict(x_test_pad, verbose=0)
    # print(preds[0])     # [0.46767616]


# review_model_origin()
review_model_embedding()


# review_model_origin
# acc: 0.71

# review_model_embedding
# acc: 0.7052
