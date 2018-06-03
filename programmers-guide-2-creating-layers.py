import tensorflow as tf

sess = tf.Session()
'''
x = tf.placeholder(tf.float32, shape=[None, 3])

linear_model = tf.layers.Dense(units=1)

y = linear_model(x)

init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))

'''
'''
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess.run((var_init, table_init))

print(sess.run(inputs))
'''
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units = 1)

y_pred = linear_model(x)

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))



loss = tf.losses.mean_squared_error(labels=y_true, predictions= y_pred)
print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
writer = tf.summary.FileWriter(".", sess.graph)

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)
print(sess.run(y_pred))
writer.close()
