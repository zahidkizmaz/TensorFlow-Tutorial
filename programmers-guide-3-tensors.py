import tensorflow as tf 
#Passing a single number, however, returns a subvector of a matrix, as follows:
'''
my_row_vector = my_matrix[2]
my_column_vector = my_matrix[:, 3]
'''
#The : notation is python slicing syntax for "leave this dimension alone". This is useful in higher-rank Tensors, as it allows you to access its subvectors, submatrices, and even other subtensors.

rank_three_tensor = tf.ones([3,4,5])
matrix = tf.reshape(rank_three_tensor, [6,10])
matrixB = tf.reshape(matrix,[3,-1])# -1 tells reshape to calculate the size of this dimension

with tf.Session() as sess:
    print(sess.run((matrix, matrixB)))
    matrix += 1
    print(matrix.eval())
   # tf.Session(config=tf.ConfigProto(log_device_placement=True))get available device list
    writer = tf.summary.FileWriter(".", sess.graph)
    
writer.close()
