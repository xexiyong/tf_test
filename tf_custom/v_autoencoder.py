import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from matplotlib import pyplot as plt

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_image(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


h_dim = 20
batch_size = 512
learning_rate = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size * 5).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size)

print(x_train.shape, x_test.shape)

z_dim = 10

class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()
        # encoder
        # self.encoder = keras.Sequential(
        #     [
        #         layers.Dense(256, activation=tf.nn.relu),
        #         layers.Dense(128, activation=tf.nn.relu),
        #         layers.Dense(h_dim, ),
        #     ]
        # )
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)

        # decoder
        # self.decoder = keras.Sequential([
        #     layers.Dense(128, activation=tf.nn.relu),
        #     layers.Dense(256, activation=tf.nn.relu),
        #     layers.Dense(784),
        # ])
        self.fc4 = layers.Dense(128,)
        self.fc5 = layers.Dense(784)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def decode(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, var):
        eps = tf.random.normal(var.shape)
        std = tf.exp(var)**0.5

        z = mu + std * eps

        return z

    def call(self, inputs, training=None):
        # b, 784--> b, z_dim
        mu, var = self.encode(inputs)

        z = self.reparameterize(mu, var)
        # b, 10 --> b, 784
        x_hat = self.decode(z)

        return x_hat, mu, var


model = VAE()
model.build(input_shape=(4, 784))
model.summary()
optim = keras.optimizers.Adam(learning_rate)

for epoch in range(100):
    for step, x in enumerate(train_db):
        # b, 28, 28
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, var = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

            kl_divergence = - 0.5 * (var + 1 - mu**2 - tf.exp(var))
            kl_divergence = tf.reduce_mean(kl_divergence) / x.shape[0]

            loss = rec_loss + 1. * kl_divergence

        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss), float(loss))

    # x = next(iter(test_db))
    # # x = tf.reshape(x, [-1, 784])
    # logits = model(tf.reshape(x, [-1, 784]))
    # x_hat = tf.nn.sigmoid(logits)
    #
    # x_hat = tf.reshape(x_hat, [-1, 28, 28])
    #
    # x_concat = tf.concat([x, x_hat], axis=0)
    # x_concat = x_concat.numpy() * 255.
    # x_concat = x_concat.astype(np.uint8)
    #
    # save_image(x_concat, 'ae_images/rec_epoch_%d.png')

    z = tf.random.normal((batch_size, z_dim))
    logits = model.decode(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)

    save_image(x_hat, 'vae_images/sample_epoch%d.png'%epoch)