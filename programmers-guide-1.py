import tensorflow as tf

logs_path = '/home/zahid/Desktop/tensorflow-tut/'

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)

total = a+b

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

sess = tf.Session()

#print(sess.run(total))

#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
'''
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2


print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
'''

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

#print(sess.run(z, feed_dict={x:3,y:4.5}))
#print(sess.run(z, feed_dict={x:[1,3],y:[2,4]}))

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

print('-----------------------------------')

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)

while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break
    
