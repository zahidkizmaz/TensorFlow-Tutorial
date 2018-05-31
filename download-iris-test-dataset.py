import tensorflow as tf
import os 


test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

print("Local copy of the dataset file: {}".format(test_fp))
