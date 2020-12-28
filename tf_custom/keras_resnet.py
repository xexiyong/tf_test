import tensorflow as tf
from tensorflow.keras import datasets, optimizers, Sequential, metrics, layers
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


class ClassicBlock(layers.Layer):

    def __init__(self, filter_num, strides=1, ):
        super(ClassicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), 1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides, ))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        ])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(100, )

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, block_num, stride=1):
        res_blocks = Sequential()
        res_blocks.add(ClassicBlock(filter_num, stride))

        for _ in range(1, block_num):
            res_blocks.add(ClassicBlock(filter_num, 1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([2, 3, 4, 6])





def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(64)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    optim = optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True);
                loss = tf.reduce_mean(loss, )

            grads = tape.gradient(loss, model.trainable_variables )
            optim.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss))

        correct_num, global_num = 0, 0
        for x, y in test_db:
            logits = model(x)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            correct_num += int(correct)
            global_num += x.shape[0]

        print(epoch, 'acc:', correct_num / global_num)


if __name__ == '__main__':
    main()
