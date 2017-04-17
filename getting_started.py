import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
  result = sess.run(product)
  print(result)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

with tf.Session() as sess:
  result = sess.run([node1, node2])
  print(result)

node3 = tf.add(node1, node2)
sess = tf.Session()
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

