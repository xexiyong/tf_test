import tensorflow as tf
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1024)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices    # 10*6  -->  10*6
    print('[accuracy] pred: ', pred)

    pred = tf.transpose(pred, perm=[1, 0])        # 6*10
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)
    print('[accuracy] target_ ', target_, ' correct: ', correct)

    res = list()
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * 100 / batch_size)
        res.append(acc)

    return res


output = tf.random.normal([10, 6])   # 10 6
output = tf.math.softmax(output, axis=1)  # 10 6
target = tf.random.uniform([10], maxval=6, dtype=tf.int32) # 10

print('prob: ', output.numpy(), ' topk_: ', tf.math.top_k(output, 3))

pred = tf.argmax(output, axis=1)
print('pred: ', pred.numpy())
print('label: ', target.numpy())

acc = accuracy(output, target, topk=(1,2,3,4,5,6))
print('top 1--6 acc: ', acc)

