import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def regression_basic():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100)
    print(model.evaluate(x, y))

    preds = model.predict(x)
    print(preds)


def regression_cars():
    def get_cars():
        cars = pd.read_csv('data/cars.csv', index_col=0)
        print(cars)

        return cars.speed.values, cars.dist.values

    x, y = get_cars()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100)
    print(model.evaluate(x, y))

    preds = model.predict([0, 30])
    print(preds)

    plt.plot(x, y, 'ro')
    plt.plot([0, 30], preds.reshape(-1))
    plt.show()


def regression_trees():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    x, y = np.float32(trees.values[:, :-1]), trees.values[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100)
    print(model.evaluate(x, y))


# regression_basic()
# regression_cars()
regression_trees()
