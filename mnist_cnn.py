import tensorflow as tf
#import random
import matplotlib.pyplot as plt
import numpy as np

# this is for comparing the accuracies
#random.seed(627)
tf.set_random_seed(627)

# get input data from tensorflow mnist module in one_hot method
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# mnist input data size is 28 * 28
# there are 10 categories(0-9)
# keep_prob is a variable for dropout
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# what is stddev?
# why the number of filters is 32?
# change into xavier initializer
with tf.name_scope('layer1'):
	#W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
	W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
	# convolution
	L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
	# relu
	L1 = tf.nn.relu(L1)
	# max pooling
	L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('layer2'):
	#W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
	W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
	L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
	L2 = tf.nn.relu(L2)
	L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('fully_connected_layer'):
	#W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
	W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 128], initializer=tf.contrib.layers.xavier_initializer())
	L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
	L3 = tf.matmul(L3, W3)
	L3 = tf.nn.relu(L3)
	L3 = tf.nn.dropout(L3, keep_prob)

with tf.name_scope('last_layer'):
	#W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
	W4 = tf.get_variable("W4", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
	model = tf.matmul(L3, W4)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# try with other optimizers and change learning rate
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# before starting training, initialize variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# set batch size
# change batch size
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# tensorboard --logdir=./logs --port=any number you want
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

# training start
# change epoch
for epoch in range(2):
	total_cost = 0

	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape(-1, 28, 28, 1)

		# change keep_prob
		_, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

		total_cost += cost_val

		summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
		writer.add_summary(summary, i)

	print('Epoch:', epoch + 1, '	Average cost =', total_cost / total_batch)
# end of training

# calculate the accuracy using test set
print('Accuracy =', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob: 1}))

# use matplot
labels = sess.run(model, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob: 1})
fig = plt.figure()
for i in range(10):
	subplot = fig.add_subplot(2, 5, i + 1)
	subplot.set_xticks([])
	subplot.set_yticks([])
	subplot.set_title(np.argmax(labels[i]))
	subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)
plt.show()
