# review_NaverMovie
# sentimental_classification

import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def get_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', index_col=0)
    df = df.dropna()
    # df.info()

    x, y = [], []
    for row in zip(df.document.values, df.label.values):
        x.append(clean_str(row[0]))
        y.append(row[1])

    return np.array(x[:2000]), np.array(y[:2000])


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


def show_token_counts(doc_tokens):
    counts = [len(tokens) for tokens in doc_tokens]
    counts = sorted(counts)

    plt.plot(counts)
    plt.show()


def make_vocab(doc_tokens, vocab_size=2000):
    freq = nltk.FreqDist([t for tokens in doc_tokens for t in tokens.split(' ')])
    # print(freq.most_common(vocab_size))
    # for t, _ in freq.most_common(vocab_size):
    #     print(t)
    return [t for t, _ in freq.most_common(vocab_size)]


def make_feature(tokens, vocab):
    feature, tokens = {}, set(tokens)
    for ch in vocab:
        feature[f'has_{ch}'] = (ch in tokens)

    # print(feature)          # {'has_!': True, 'has_,': True, 'has_영화': True, ...}
    return feature


def make_feature_data(doc_tokens, labels, vocab, feature_func):
    # for tokens, label in zip(doc_tokens, labels):
    #     print(tokens, label)        # 아 더빙 진짜 짜증나네요 목소리 0
    return [(feature_func(tokens.split(' '), vocab), label) for tokens, label in zip(doc_tokens, labels)]


def make_feature_1(doc_tokens, vocab):
    feature = []
    for tokens in doc_tokens:
        tokens = set(tokens.split(' '))
        # print(tokens)       # {'더빙', '아', '짜증나네요', '목소리', '진짜'}
        feature.append([v in tokens for v in vocab])
        # print(feature)

    return np.float32(feature)


def make_feature_2(docs, vocab, pad_size=20):
    lb = preprocessing.LabelBinarizer()
    lb.fit(vocab)

    transformed = []
    for tokens in docs:
        new_tokens = tokens.split(' ') + ['*'] * max(pad_size - len(tokens.split(' ')), 0)
        # print(new_tokens)       # ['아', '더빙', '진짜', '짜증나네요', '목소리', '*', '*', ..., '*']
        new_tokens = new_tokens[:pad_size]
        new_tokens = lb.transform(new_tokens)
        transformed.append(new_tokens)

    transformed = np.float32(transformed)
    # transformed = transformed.reshape(transformed.shape[0], -1)
    # print(transformed.shape)        # (2000, 40000)
    return transformed.reshape(transformed.shape[0], -1)


def sentimental_classification():
    x_train, y_train = get_data('data/naver_ratings_train.txt')
    x_test, y_test = get_data('data/naver_ratings_test.txt')
    # print(x_train.shape, y_train.shape)                 # (149995,) (149995,)
    # print(x_test.shape, y_test.shape)                   # (49997,) (49997,)

    # show_token_counts(x_train)
    vocab = make_vocab(x_train)

    train_set = make_feature_data(x_train, y_train, vocab, make_feature)
    test_set = make_feature_data(x_test, y_test, vocab, make_feature)

    # print([make_feature(tokens.split(' '), vocab) for tokens in x_train])
    # print(train_set[0])

    clf = nltk.NaiveBayesClassifier.train(train_set)
    print(clf.classify(make_feature(x_test[17].split(' '), vocab)))
    dist = clf.prob_classify(make_feature(x_test[17].split(' '), vocab))
    print(list(dist.samples()))     # [0, 1]
    print(dist.prob(0))             # 0.8467593602543887
    print(dist.prob(1))             # 0.1532406397456092
    # print('acc: ', nltk.classify.accuracy(clf, test_set))
    # clf.show_most_informative_features()

    # ------------------------------------------------------------------------- #

    x_train, y_train = make_feature_1(x_train, vocab), np.float32(np.reshape(y_train, newshape=[-1, 1]))
    x_test, y_test = make_feature_1(x_test, vocab), np.float32(np.reshape(y_test, newshape=[-1, 1]))

    # x_train, y_train = make_feature_2(x_train, vocab), np.float32(np.reshape(y_train, newshape=[-1, 1]))
    # x_test, y_test = make_feature_2(x_test, vocab), np.float32(np.reshape(y_test, newshape=[-1, 1]))

    # print(x_train.shape, x_test.shape)      # (2000, 2000) (2000, 2000)
    # print(y_train.shape, y_test.shape)      # (2000, 1) (2000, 1)

    w = tf.Variable(tf.zeros([x_train.shape[1], 1]))
    b = tf.Variable(tf.zeros([1]))

    z = tf.matmul(x_train, w) + b
    hx = tf.nn.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y_train)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(tf.nn.sigmoid(tf.matmul(x_test, w) + b))
    preds_bool = (preds > 0.5)

    print('acc: ', np.mean(preds_bool == y_test))
    # print(preds[9])
    sess.close()


sentimental_classification()

# nltk.NaiveBayesClassifier
# acc:  0.702

# make_feature_1
# acc:  0.6765

# make_feature_2
# lr = 0.01
# acc:  0.5925

# make_feature_2
# lr = 0.1
# acc:  0.603
