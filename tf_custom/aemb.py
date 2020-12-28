import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1001, 64, input_length=3, mask_zero=True))
# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch
# dimension.
input_array = np.random.randint(1000, size=(32, 3))
input_array[0,2] = 0
input_array[0,1] = 0
print(input_array.shape, input_array)

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape, output_array[0])

# 32, 10 --> 32, 10, 64
