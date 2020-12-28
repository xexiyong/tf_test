import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_words = 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
max_review_length = 80
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)

batchsize = 64
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsize, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsize, drop_remainder=True)
print('x_train shape: ', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape: ', x_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        #
        self.state0 = [tf.zeros([batchsize, units]), tf.zeros([batchsize, units])]
        self.state1 = [tf.zeros([batchsize, units]), tf.zeros([batchsize, units])]
        # [b, 80] --> [b, 80, 100]
        self.embedding = layers.Embedding(total_words, 100, input_length=max_review_length)

        # self.rnn_cell0 = layers.LSTMCell(units, dropout=0.5)
        # self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)
        # self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        # self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn = keras.Sequential([
            layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.5, unroll=True),
        ])

        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        # inputs: b, 80
        x = inputs
        # b, 80 --> b, 80, 100
        x = self.embedding(x)


        # state0 = self.state0
        # state1 = self.state1
        # # b, 80, 100  -> b, 64.  second axis iterate. so unstack.
        # for word in tf.unstack(x, axis=1):  # b, 100
        #     # h1 = x * wxh + whh
        #     out0, state0 = self.rnn_cell0(word, state0, training)
        #     # state0 = state1
        #     out1, state1 = self.rnn_cell1(out0, state1, training)


        out1 = self.rnn(x)


        # out: [b, 64] --> b, 1
        x = self.fc(out1)
        # p = a(
        prob = tf.sigmoid(x)

        return prob


def main():
    unit = 64
    epoches = 4

    model = MyRNN(unit)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(db_train, epochs=epoches, validation_data=db_test)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()