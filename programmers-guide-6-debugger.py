#need to add this code to the program for debug feature


if __name__ = "__main__":
	tf.app.run()
	
from tensorflow.python import debug as tf_debug

sess = tf_debug.LocalCLIDebugWrapperSession(sess)

