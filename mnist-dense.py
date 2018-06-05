import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

#model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1,784]), W) + b)

#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss func
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# is correct
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(1000):
		#load batch of images and correct answers
		batch_X, batch_Y = mnist.train.next_batch(100)
		train_data = {X: batch_X, Y_:batch_Y}

		#train
		sess.run(train_step, feed_dict=train_data)
		
		#success?
		a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

		#sucess on test data?
		test_data = {X: mnist.test.images, Y_: mnist.test.labels}
		a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
		print("Accuracy:",a)
		print("C:",c)
		






