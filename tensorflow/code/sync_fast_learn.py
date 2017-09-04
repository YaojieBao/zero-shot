import sys
import numpy as np
import random
import tensorflow as tf

def generate_batch(x, idx, n, batch_size):
    batch_index = random.sample(xrange(n), batch_size)
    return x[batch_index], idx[batch_index]

def learning(Xbase, Ybase, lamda):
    print '''-----run tensorflow start-----'''
    # parameters
    #batch_size = 19443
    learning_rate = 0.1
    training_epochs = 150 # 150
    display_step = 1

    # set session
    sess = tf.Session()

    # get sizes
    sample_num, image_size = Xbase.shape
    # Ybase = np.array([1,2,0,3,2]) # test ybase
    labels = np.unique(Ybase)
    class_num = labels.shape[0]

    # get label index
    ind = -np.ones((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[Ybase == labels[i], i] = 1;

    # input image
    X = tf.placeholder(tf.float32, shape=[None, image_size])
    W = tf.Variable(tf.random_normal([class_num, image_size]))
    #W = tf.get_variable(shape=[class_num, image_size], initializer=tf.contrib.layers.xavier_initializer())
    XW = tf.matmul(X, tf.transpose(W));

    # loss function
    IND = tf.placeholder(tf.float32, shape=[None, class_num])
    L = tf.square(tf.maximum(0.0, 1 - tf.multiply(IND, XW)))
    cost = tf.reduce_sum(L) / sample_num / class_num + lamda / 2 * tf.reduce_sum(tf.square(W));

    # set optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # init variables
    init = tf.global_variables_initializer()

    # start
    sess.run(init)
    #total_batch = int(sample_num/batch_size);

    # Training cycle
    for epoch in range(training_epochs):
	# Run optimization op (backprop) and cost op (to get loss value)
        _, loss = sess.run([optimizer, cost], feed_dict={X: Xbase, IND: ind})
	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss))

    # output W
    W_opt = sess.run(W)
    print '''-----run tensorflow end------'''
    return W_opt
