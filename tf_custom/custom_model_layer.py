import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, models
from tensorflow import keras


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y


batchsize = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets: ', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsize)
ds_val = tf.data.Dataset.from_tensor_slices(((x_val, y_val)))
ds_val = ds_val.map(preprocess).batch(batchsize)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

# network = Sequential([
#     layers.Dense(256, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10, ),
# ])
# network.build(input_shape=(None, 28 * 28))
# network.summary()


class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def build(self, input_shape):
        super(MyModel, self).build(input_shape)
        return

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
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
network.compile(optimizer=optimizers.Adam(lr=0.01), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.build(input_shape=(None, 28 * 28))
network.summary()

network.fit(db, epochs=1, validation_data=ds_val, validation_freq=1)
network.evaluate(ds_val)

# sample = next(iter(ds_val))
# x = sample[0]
# y = sample[1]
# pred = network.predict(x)
# y = tf.argmax(y, axis=1)
# pred = tf.argmax(pred, axis=1)
#
# print(pred)
# print(y)

# way 1 save load model weights.
# network.save_weights('weights.ckpt')
# del network
#
# network = MyModel()
# network.compile(optimizer=optimizers.Adam(lr=0.01), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# network.load_weights('weights.ckpt')
# network.evaluate(ds_val)


# way 2 save h5. no support subclass customized model. function model+sequential model is allowed.
# network.save('model.tf', save_format="tf")  same as way 3
# models.save_model(network, 'model.h5',)
# del network
#
# network = models.load_model('model.h5')
# network.evaluate(ds_val)



# way 3 more general other langs.
# tf.saved_model.save(network, 'model_dumps/')
# del network
#
# imported = tf.saved_model.load('model_dumps/')
# f = imported.signatures['serving_default']
# # print(f)
# ds_output = f(sample[0])
# print(ds_output)
