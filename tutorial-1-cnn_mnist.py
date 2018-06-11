import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)

#mnist = mnistdata.read_data_sets("MNIST_data", one_hot=True, reshape=False)
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]

# input X is the images pixels by 28x28, first parameter (None) is the index of image in mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28 ,1])

# correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# step var for decreasing learning rate
step = tf.placeholder(tf.int32)

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev = 0.1))
B1 = tf.Variable(tf.ones([4])/10)

W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev = 0.1))
B2 = tf.Variable(tf.ones([8])/10)

W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev = 0.1))
B3 = tf.Variable(tf.ones([12])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev = 0.1))
B4 = tf.Variable(tf.ones([200])/10)

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev = 0.1))
B5 = tf.Variable(tf.ones([10])/10)

# Model
stride = 1 # output is 28x28
conv_layer1 = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') 
Y1 = tf.nn.relu(conv_layer1 + B1)

stride = 2 # output is 14x14
conv_layer2 = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') 
Y2 = tf.nn.relu(conv_layer2 + B2)

stride = 2 # output is 7x7
conv_layer3 = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') 
Y3 = tf.nn.relu(conv_layer3 + B3)

# reshape the output matrix from the third conv layer for the fully connected
YY = tf.reshape(Y3, shape=[-1, 7*7*12])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})
    print('Training step:',i)
for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)
