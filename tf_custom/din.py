import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

x = {
    'user': np.array([0, 1, 2]),
    'gender': np.array([0, 1, 0]),
    'item': np.array([1, 2, 3]),
    'item_gender': np.array([1, 2, 1]),
    'score': np.array([0.1, 0.2, 0.3]),

    'hist_item': np.array([[1, 2, 3, 0],
                           [1, 2, 3, 0],
                           [1, 2, 0, 0]]),
    'hist_item_gender': np.array([[1, 1, 2, 0],
                                  [2, 1, 1, 0],
                                  [2, 1, 0, 0]])
}
y = np.array([1, 0, 1])
user_input = tf.keras.Input((1,), name='user', dtype=tf.int32)
gender_input = tf.keras.Input((1), name='gender', dtype=tf.int32)
item_input = tf.keras.Input((1), name='item', dtype=tf.int32)
item_gender_input = tf.keras.Input((1,), name='item_gender', dtype=tf.int32)
score_input = tf.keras.Input((1,), name='score', dtype=tf.float32)

hist_item_input = tf.keras.Input((4,), name='hist_item', dtype=tf.int32)
hist_item_gender_input = tf.keras.Input((4,), name='hist_item_gender', dtype=tf.int32)

inputs_list = [user_input, gender_input, item_input, item_gender_input, score_input, hist_item_input,
               hist_item_gender_input]

# l2_reg = 1e-5
l2_reg_embedding = 1e-5
l2_reg_dnn = 1e-5
seed = 2021
dnn_dropout = 0.5
dnn_activation = 'relu'
dnn_hidden_units = [5, 4, 3, 2]
dnn_embedding_size = 4
att_hidden_size = (80, 40)
att_activation = "dice"
att_weight_normalization = False
dnn_use_bn = False

user_emb = layers.Embedding(3, 4, embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=0.0001,
                                                                                            seed=2020),
                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                            name='user_emb')
gender_emb = layers.Embedding(2, 4, embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                              stddev=0.0001,
                                                                                              seed=2020),
                              embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                              name='gender_emb')
item_emb = layers.Embedding(4, 4, embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                            stddev=0.0001,
                                                                                            seed=2020),
                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                            name='item_emb', mask_zero=True)
item_gender_emb = layers.Embedding(3, 4, embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                   stddev=0.0001,
                                                                                                   seed=2020),
                                   embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
                                   name='item_gender_emb', mask_zero=True)
item_query_input = item_emb(item_input)
item_gender_query_input = item_gender_emb(item_gender_input)
query_emb_list = [item_query_input, item_gender_query_input]

item_key_input = item_emb(hist_item_input)
item_gender_key_input = item_gender_emb(hist_item_gender_input)
key_emb_list = [item_key_input, item_gender_key_input]

user_emb_input = user_emb(user_input)
gender_emb_input = gender_emb(gender_input)
item_emb_input = item_emb(item_input)
item_gender_emb_input = item_gender_emb(item_gender_input)
dnn_input_emb_list = [user_emb_input, gender_emb_input, item_emb_input, item_gender_emb_input]

print('query emb list: ', query_emb_list, ' key emb list: ', key_emb_list, 'general input: ',
      dnn_input_emb_list)

# dense_input_list = score_input

keys_emb = tf.keras.layers.Concatenate()(key_emb_list)
query_emb = tf.keras.layers.Concatenate()(query_emb_list)
deep_input_emb = tf.keras.layers.Concatenate()(dnn_input_emb_list)

# None, 4, 8         None, 1, 8             None, 1, 16
print('after concat: ', keys_emb, query_emb, deep_input_emb)


# attention.
class Dice(layers.Layer):

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],),
                                      initializer=tf.keras.initializers.Zeros(),
                                      dtype=tf.float32,
                                      name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, layers.Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


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

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

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


class LocalActivationUnit(layers.Layer):

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        size = 4 * int(input_shape[0][-1]) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=tf.keras.initializers.glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

        self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(x[0], x[1], axes=(-1, 0)), x[2]))

        super(LocalActivationUnit, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        # [None, 1, 8  None, 4, 8]         [None, 1    None, 4]
        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = tf.keras.backend.repeat_elements(query, keys_len, 1)  # None, 1, 8 --> None, 4, 8

        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # None, 4, 32
        # None, 4, 32 --> None, 4, 64 --> None, 4, 32 with dice activation.
        att_out = self.dnn(att_input, training=training)
        # None, 4, 32 --> None, 4, 1
        attention_score = self.dense([att_out, self.kernel, self.bias])

        return attention_score


class AttentionSequencePoolingLayer(layers.Layer):

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        self.local_attention = LocalActivationUnit(self.att_hidden_units, self.att_activation,
                                                   l2_reg=0, dropout_rate=0, use_bn=False, seed=1024)
        super(AttentionSequencePoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):
        # [None, 1, 8  None, 4, 8]         [None, 1    None, 4]
        # print('[Attention layer] call: {},---, {}'.format(inputs, mask))
        if mask is None:
            raise ValueError("When supports_masking=True,input must support masking")
        queries, keys = inputs
        attention_score = self.local_attention([queries, keys], training=training)
        # None, 4, 1 --> None, 1, 4
        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        key_masks = tf.expand_dims(mask[-1], axis=1)  # None, 4 --> None, 1, 4
        outputs = tf.where(key_masks, outputs, paddings)  # None, 1, 4

        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)  # None, 1, 8

        outputs._uses_learning_phase = training is not None
        return outputs


# None, 1, 8
hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                     weight_normalization=att_weight_normalization,
                                     supports_masking=True)([query_emb, keys_emb])
# None, 1, 16+8
deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb, hist])
# None, 16+8
deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
# None, 16 + 8 + 1
dnn_input = tf.keras.layers.Concatenate()([deep_input_emb, score_input])

output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
final_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(
    output)


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

model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

print('model: ', model)

model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
model.fit(x, y, batch_size=100, epochs=20, validation_split=0.5)
model.summary()

# tf.keras.models.save_model(model, 'wdl.h5')
model.save_weights('din/weights.ckpt')
