import tensorflow as tf
from tensorflow.keras import datasets, optimizers, Sequential, metrics
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = 2 * (tf.cast(x, dtype=tf.float32) / 255. - 0.5)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsize = 128
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print('dataset: ', x.shape, y.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsize)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsize)

sample = next(iter(train_db))
print(sample[0].shape)


class MyDense(keras.layers.Layer):
    def __init__(self, input_dim,  output_dim, activation=None):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w1', [input_dim, output_dim])
        # self.bias = self.add_variable('b1', [output_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


class MyModel(keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)

        return x


network = MyModel()
network.compile(optimizer=optimizers.Adam(lr=0.001), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)

network.evaluate(test_db)

network.save_weights('ckpt/weights.ckpt')

del network

network = MyModel()
network.compile(optimizer=optimizers.Adam(lr=0.001), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
