import tensorflow as tf

logs_path = '/home/zahid/Desktop/tensorflow-tut/'

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)

total = a+b

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

sess = tf.Session()

print(sess.run(total))

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

