import sys
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import load_data

def generate_batch(data, label, n, batch_size):
    batch_index = random.sample(xrange(n), batch_size)
    return data[batch_index], label[batch_index]

def main():
    print "load data"
    train_x, train_y, test_x, test_y = load_data.load()
    num_inputs = len(train_x)
    print len(train_x), len(train_y), len(test_x), len(test_y)

    sess = tf.Session()

    # set parameters
    learning_rate = 0.01
    training_epochs = 200
    batch_size = 512
    display_step = 1

    # set nets
    image_size = len(train_x[0])
    hidden_size = 512
    output_size = len(train_y[0])
    print image_size, hidden_size, output_size

    train_inputs = tf.placeholder("float", shape=[None, image_size])
    train_outputs = tf.placeholder("float", shape=[None, output_size])
    W1 = tf.Variable(tf.random_normal([image_size, hidden_size]))
    b1 = tf.Variable(tf.random_normal([hidden_size]))
    h1 = tf.nn.sigmoid(tf.matmul(train_inputs, W1) + b1)

    W2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
    b2 = tf.Variable(tf.random_normal([output_size]))
    h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

    # cost function
    cost = tf.nn.l2_loss(tf.subtract(h2 , train_outputs))

    # set optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # init variables
    init = tf.global_variables_initializer()

    # start
    sess.run(init)
    total_batch = int(num_inputs/batch_size);

    # Training cycle
    for epoch in range(training_epochs):
	# Loop over all batches
	for i in range(total_batch):
	    batch_x, batch_y = generate_batch(train_x, train_y, num_inputs, batch_size)
	    # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={train_inputs: batch_x, train_outputs: batch_y})
	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    # run test data
    test_output = sess.run(h2, feed_dict={train_inputs: test_x})

    # test accuracy
    y_predict = [np.argmax(i)+1 for i in test_output]
    y_true = [np.argmax(i)+1 for i in test_y]

    print len(y_predict), len(y_true)
    print "test accuracy: ", accuracy_score(y_predict, y_true)

    # seen and unseen accuracy
    seen_classes = [1, 2, 3, 5, 6, 7, 8, 9]
    unseen_classes = [4, 10]
    seen_right = 0;
    seen_total = 0;
    unseen_right = 0;
    unseen_total = 0;
    for i in range(len(y_true)):
        predict = y_predict[i]
        true = y_true[i]
        if (true != 4 and true != 8): # seen classes
            seen_total = seen_total + 1;
            if (predict == true):
                seen_right = seen_right + 1;
        else:
            unseen_total = unseen_total + 1;
            if (predict == true):
                unseen_right = unseen_right + 1;
    print "seen total: ", seen_total, " seen_right: ", seen_right, " seen_acc_per: ", float(seen_right)/seen_total
    print "unseen total: ", unseen_total, " unseen_right: ", unseen_right, " unseen_acc_per: ", float(unseen_right)/unseen_total


if __name__ == "__main__":
    main()
