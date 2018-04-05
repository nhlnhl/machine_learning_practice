import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

image_size = 28
weight_size = 4
num_filters1 = 16
num_filters2 = 32
num_layer3 = 256
num_classes = 10

keep_prob = 0.8
learning_rate = 0.001
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
Y = tf.placeholder(tf.float32, [None, num_classes])

def conv_pool_layer(X, W):
	return tf.nn.max_pool(tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME'), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fully_connected_layer(X, W, keep_prob):
	return tf.nn.dropout(tf.matmul(X, W), keep_prob)

W1 = tf.truncated_normal([weight_size, weight_size, 1, num_filters1], stddev=0.1)
L1 = conv_pool_layer(X, W1)

W2 = tf.truncated_normal([weight_size, weight_size, num_filters1, num_filters2], stddev=0.1)
L2 = conv_pool_layer(L1, W2)

W3 = tf.truncated_normal([int(image_size * image_size * num_filters2 / 16), num_layer3], stddev=0.1)
L3 = tf.reshape(L2, [-1, int(image_size * image_size * num_filters2 / 16)])
L3 = fully_connected_layer(L3, W3, keep_prob)

W4 = tf.truncated_normal([num_layer3, num_classes], stddev=0.1)
model = fully_connected_layer(L3, W4, 1)

cost = tf.Variable(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	for epoch in range(10):
		total_cost = 0

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			batch_xs = batch_xs.reshape(-1, image_size, image_size, 1)
			_, batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})

			total_cost += batch_cost

		print("Epoch:", epoch + 1, "Cost:", total_cost)
