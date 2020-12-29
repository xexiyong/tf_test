# from models/deepfm.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from collections import namedtuple, OrderedDict, defaultdict

y = np.array([1, 0, 0, 0, 0, 0, 1, 0])
x = {
    'sparse_feature_0': np.array([4, 4, 0, 3, 1, 1, 3, 2]),
    'sparse_feature_1': np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    'dense_feature_0': np.array([0.31088728, 0.69594661, 0.33354281, 0.60095433, 0.58161671,
                                 0.04908458, 0.58981453, 0.98049298]),
    'dense_feature_1': np.array([0.15422162, 0.17702625, 0.56617806, 0.87792887, 0.11278011,
                                 0.51292355, 0.16292026, 0.7452631]),
}

# -------------------------feature------------------------------------
input_features = OrderedDict()
input_features['sparse_feature_0'] = layers.Input(shape=(1), name='sparse_feature_0', dtype=tf.int32)
input_features['sparse_feature_1'] = layers.Input(shape=(1), name='sparse_feature_1', dtype=tf.int32)
input_features['dense_feature_0'] = layers.Input(shape=(1), name='dense_feature_0', dtype=tf.float32)
input_features['dense_feature_1'] = layers.Input(shape=(1), name='dense_feature_1', dtype=tf.float32)

input_list = list(input_features.values())  # keras.model.Model's input.

l2_reg = 1e-5
l2_reg_embedding = 1e-5
l2_reg_dnn = 1e-5
seed = 2021
dnn_dropout = 0.5
dnn_activation = 'relu'
dnn_hidden_units = [5, 4, 3, 2]
dnn_embedding_size = 4
# -------------------linear----------------------------------
sparse_feature_0_emb = layers.Embedding(6, 1,
                                        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                  stddev=0.0001,
                                                                                                  seed=2020),
                                        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
                                        name='sparse_feature_0_emb')

sparse_feature_1_emb = layers.Embedding(2, 1,
                                        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                  stddev=0.0001,
                                                                                                  seed=2020),
                                        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
                                        name='sparse_feature_1_emb')

sparse_input = layers.Concatenate(axis=-1)(
    [sparse_feature_0_emb(input_features['sparse_feature_0']),
     sparse_feature_1_emb(input_features['sparse_feature_1'])])
dense_input = layers.Concatenate(axis=-1)(
    [input_features['dense_feature_0'],
     input_features['dense_feature_1']])

print('sparse input: ', sparse_input, 'dense input: ', dense_input)


class Linear(layers.Layer):

    def __init__(self, l2_reg=0.0, use_bias=True, seed=1024, **kwargs):

        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        self.kernel = self.add_weight(
            'linear_kernel',
            shape=[int(input_shape[1][-1]), 1],
            initializer=tf.keras.initializers.glorot_normal(self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True)

        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sparse_input, dense_input = inputs
        fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
        linear_logit = tf.reduce_sum(sparse_input, axis=-1) + fc
        if self.use_bias:
            linear_logit += self.bias
        return linear_logit


linear_logit = Linear(l2_reg, seed=1024)([sparse_input, dense_input])

print('linear logits: ', linear_logit)

# ---------------------fm cross-------------------------------

deep_sparse_feature_0_emb = layers.Embedding(6, dnn_embedding_size,
                                             embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                       stddev=0.0001,
                                                                                                       seed=2020),
                                             embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                                             name='deep_sparse_feature_0_emb')
deep_sparse_feature_1_emb = layers.Embedding(2, dnn_embedding_size,
                                             embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                       stddev=0.0001,
                                                                                                       seed=2020),
                                             embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                                             name='deep_sparse_feature_1_emb')

fm_cross_feature_0 = deep_sparse_feature_0_emb(input_features['sparse_feature_0'])  # (None, 1, k) size
fm_cross_feature_1 = deep_sparse_feature_1_emb(input_features['sparse_feature_1'])

fm_cross_feature_matrix = layers.Concatenate(axis=1)([fm_cross_feature_0, fm_cross_feature_1])  # None, 2, k
fm_cross_feature_term1 = tf.reduce_sum(fm_cross_feature_matrix, axis=1, keepdims=True)  # None, 1, k
fm_cross_feature_term1 = tf.square(fm_cross_feature_term1)

fm_cross_feature_term2 = tf.square(fm_cross_feature_matrix)
fm_cross_feature_term2 = tf.reduce_sum(fm_cross_feature_term2, axis=1, keepdims=True)  # None, 1, k

fm_logit = (fm_cross_feature_term1 - fm_cross_feature_term2) * 0.5  # None, 1, k
fm_logit = tf.reduce_sum(fm_logit, axis=2)

print('fm : ', fm_cross_feature_matrix, fm_logit)

# ---------------------dnn------------------------------------

deep_sparse_input = layers.Concatenate(axis=-1)(
    [deep_sparse_feature_0_emb(input_features['sparse_feature_0']),
     deep_sparse_feature_1_emb(input_features['sparse_feature_1'])])
deep_dense_input = layers.Concatenate(axis=-1)(
    [input_features['dense_feature_0'],
     input_features['dense_feature_1']])

deep_sparse_input = layers.Flatten()(deep_sparse_input)
dnn_input = layers.Concatenate(axis=-1)([deep_sparse_input, deep_dense_input])

print('sparse input: ', deep_sparse_input, 'dense input: ', deep_dense_input, 'dnn input: ', dnn_input)


class DNN(layers.Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0., dropout_rate=0., use_bn=False,
                 output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=tf.keras.initializers.glorot_normal(
                                            seed=self.seed),
                                        regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [tf.keras.layers.Activation(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = tf.keras.layers.Activation(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input


dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)

dnn_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(
    dnn_out)

final_logit = layers.add([dnn_logit, linear_logit, fm_logit])


class PredictionLayer(layers.Layer):

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=tf.keras.initializers.Zeros(), name="global_bias")
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output


output = PredictionLayer('binary')(final_logit)
model = tf.keras.models.Model(inputs=input_list, outputs=output)

print('model: ', model)

model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
model.fit(x, y, batch_size=100, epochs=20, validation_split=0.5)
model.summary()

# tf.keras.models.save_model(model, 'wdl.h5')
model.save_weights('deepfm/weights.ckpt')
