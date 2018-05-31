import tensorflow as tf
import os

tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to hide the error.

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

iris_path = "/home/zahid/Desktop/tensorflow-tut/datasets/iris_training.csv"

train_dataset = tf.data.TextLineDataset(iris_path)
train_dataset = train_dataset.skip(1)
train_dataset = train_dataset.map(parse_csv)
train_dataset = train_dataset.shuffle(buffer_size = 1000)
train_dataset = train_dataset.batch(32)

features, label = iter(train_dataset).next()
print("example features: ", features[0])
print("example label: ", label[0])
