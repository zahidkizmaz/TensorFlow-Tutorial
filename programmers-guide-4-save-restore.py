import tensorflow as tf

#create vars
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

#add an op to init vars
init_op = tf.global_variables_initializer()

#####add ops to save and restore all the variables
saver = tf.train.Saver()

#lanching the model
#itialize the varibles 
#do some work
#save the vars to disk

with tf.Session() as sess:
	sess.run(init_op)
	
	#do some work
	inc_v1.op.run()
	dec_v2.op.run()
	
	#save vars to disk
	save_path = saver.save(sess, "/home/zahid/Desktop/tensorflow-tut")
	print("Model saved in path: {}".format(save_path))
	print(inc_v1.eval())
	

# resetting graph
tf.reset_default_graph()

#create vars
# note you will find vars as it is initialized before, so no need to initialize here!
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

#####add ops to save and restore all the variables
saver = tf.train.Saver()

#lanching the model
#use the save to restore the varibles 
#do some work

with tf.Session() as sess:
	saver.restore(sess, "/home/zahid/Desktop/tensorflow-tut")
	print("model restored")

	#check the vars values

	print("v1: %s" % v1.eval())
	print("v2: %s" % v2.eval())
