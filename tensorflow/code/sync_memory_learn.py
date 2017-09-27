import sys
import os
import numpy as np
import random
import tensorflow as tf
from scipy.special import expit as sigmoid

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_batch(x, idx, n, batch_size):
    batch_index = random.sample(xrange(n), batch_size)
    return x[batch_index], idx[batch_index]

''' get accuracy '''
def evaluate_easy(Ypred, Ytrue):
    labels = np.unique(Ytrue)
    L = labels.shape[0]
    confusion = np.zeros((L, 1))
    for i in range(L):
        confusion[i] = float(np.sum(np.logical_and(Ytrue == labels[i], Ypred == labels[i]))) / np.sum(Ytrue == labels[i])
    acc = np.mean(confusion)
    acc2 = np.mean(Ypred == Ytrue)
    return acc2

def variable_summaries(var):
    with tf.name_scope('summaries'):
	mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        #stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def linear_layer(inputs, num_outputs, reuse=None, scope_postfix='', **kwargs):
    ''' Add a linear fully connected layer'''
    regularizer=tf.contrib.layers.l2_regularizer(scale=0.002)
    with tf.variable_scope("linear"+scope_postfix, reuse=reuse):
        return tf.contrib.layers.fully_connected(inputs, num_outputs,
                activation_fn=None, weights_regularizer=regularizer, **kwargs)

def learning(Sig_Y, Xbase, Ybase, lamda, Sim_base, Xval, Yval, Sim_val):
    print Sim_base

    ''' Xbase : input embedding image feature
        Ybase : input image label id
        Sig_Y : label embedding
    '''

    print '''-----run tensorflow start-----'''
    # parameters
    #batch_size = 19443
    learning_rate = 0.05
    training_epochs = 100 # 500
    display_step = 1
    x_dropout_rate = 0.9
    m_dropout_rate = 0.9

    # set session
    #with tf.device('/gpu:1'):
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    # get sizes
    #Ybase = np.array([1,2,0,3,2]) # test ybase
    sample_num, image_size = Xbase.shape # 19443 * 1024
    labels = np.unique(Ybase)
    class_num = labels.shape[0] # 32
    memory_size = Sig_Y.shape[1]
    memory_embedding_size = 100

    test_sample_num = Xval.shape[0]
    test_labels = np.unique(Yval)
    test_class_num = test_labels.shape[0] # 8

    # get label index
    #ind = -np.ones((sample_num, class_num), dtype=np.float32)
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[Ybase == labels[i], i] = 1;

    # get test index
    #ind = -np.ones((sample_num, class_num), dtype=np.float32)
    test_ind = np.zeros((test_sample_num, test_class_num), dtype=np.float32)
    print "test_labels", test_labels
    for i in range(test_class_num):
        test_ind[Yval == test_labels[i], i] = 1;

    # input image
    with tf.name_scope('input'):
        input_x = tf.placeholder(tf.float32, shape=[None, image_size]) # 19443, 1024
        X = tf.nn.dropout(input_x, x_dropout_rate)

    with tf.variable_scope('input_emb'):
        u = linear_layer(input_x, memory_embedding_size, scope_postfix='ux') # [19443, 100]
        variable_summaries(u)
        c = linear_layer(input_x, memory_embedding_size, scope_postfix='cx') # [19443, 100]
        variable_summaries(c)

    # memory lower half ===> generate pi_k, k = 32, memory class size
    with tf.name_scope('memory'):
        input_memory = Sig_Y[np.unique(Ybase-1)] # label embedding 32, 85
        memory = tf.nn.dropout(input_memory, m_dropout_rate)

    with tf.variable_scope('m'):
        m = linear_layer(memory, memory_embedding_size)
        variable_summaries(m)

    with tf.name_scope('pi_k'):
        score = tf.matmul(c, tf.transpose(m))
        pi_k = tf.nn.softmax(score) # 19443, 32
        variable_summaries(pi_k)

    # memory upper half ===>
    with tf.variable_scope('B'):
        BM_0 = linear_layer(memory, memory_embedding_size) # [32, 100]
        variable_summaries(BM_0)

    with tf.name_scope('BM'):
        BM_1 = tf.expand_dims(BM_0, 0) # [1, 32, 100]
        samples_num = tf.shape(pi_k)[0]
        BM = tf.tile(BM_1, [samples_num, 1, 1]) # [19443, 32, 100]
        variable_summaries(BM)
        BMU = tf.multiply(BM, tf.expand_dims(u, 1))
        variable_summaries(BMU) # [19443, 32, 100]

    with tf.variable_scope('Sim_base'):
        A = tf.get_variable("A", shape=[memory_embedding_size, class_num], initializer=tf.contrib.layers.xavier_initializer())
        AS = tf.matmul(A, Sim_base) #[100, 32]
        #AS = tf.transpose(linear_layer(Sim_base, memory_embedding_size)) # [100, 32]
        f_l = tf.tensordot(BMU, AS, [[2], [0]]) # [1944, 32, 32]
        variable_summaries(f_l)

    # expand dimensions for A and Sim_base
#    with tf.name_scope('A'):
#        BMUA = linear_layer(BMU, class_num)
#        variable_summaries(BMUA)

#        A_0 = tf.Variable(tf.random_normal([memory_embedding_size, class_num])) # [100, 32], mem
#        variable_summaries(A_0)
#    with tf.name_scope('A_1'):
#        A_1 = tf.expand_dims(A_0, 0) # [1, 100, 32] #    with tf.name_scope('A'):
#        A = tf.tile(A_1, [sample_num, 1, 1]) # [19443, 100, 32]
#        variable_summaries(A)
#    with tf.name_scope('BMUA'):
#        BMUA = tf.matmul(BMU, A) # [19443, 32, 32]

#    with tf.name_scope('Sim_base'):
#        Sim_base = np.tile(np.expand_dims(Sim_base, axis=0), [sample_num, 1, 1])
#        variable_summaries(Sim_base)
#
#    with tf.name_scope('f_l'):
#        f_l = tf.matmul(BMUA, Sim_base)

#        f_l_prob = tf.transpose(tf.sigmoid(f_l), [0, 2, 1]) # [19443, 32, 32]
#        variable_summaries(f_l_prob)
#    with tf.name_scope('res'):
#        res = tf.reduce_sum(tf.multiply(f_l_prob, tf.expand_dims(pi_k, 2)), 1) # [19443, 32]

    with tf.name_scope('f_l_prob'):
        f_l_prob = tf.sigmoid(f_l)
        res = tf.reduce_sum(tf.multiply(f_l_prob, tf.expand_dims(pi_k, 2)), 1)
        variable_summaries(res)

    with tf.name_scope('Ypred'):
        Ypred = tf.argmax(res, 1)
        variable_summaries(Ypred)

    # loss function
    with tf.name_scope('IND'):
        IND = tf.placeholder(tf.float32, shape=[None, class_num])
        variable_summaries(IND)
    with tf.name_scope('L'):
        #mul_ind_res = tf.multiply(IND, res)
        #L = tf.square(tf.maximum(0.0, 1 - mul_ind_res))
        L = tf.losses.softmax_cross_entropy(IND, res)
        PL = L
        #PL = tf.reduce_mean(L)
        #PL = tf.nn.l2_loss(L)
        variable_summaries(L)
    #cost = tf.reduce_sum(L) / sample_num / class_num; #original version
    with tf.name_scope('cost'):
        cost = PL # + lamda * tf.nn.l2_loss(A) + lamda * tf.nn.l2_loss(B) + lamda * tf.nn.l2_loss(U) + lamda * tf.nn.l2_loss(V)
        #cost = PL / sample_num / class_num + lamda / 2 * tf.reduce_sum(tf.square(A)) + lamda / 2 * tf.reduce_sum(tf.square(B)) + lamda / 2 * tf.reduce_sum(tf.square(U)) + lamda / 2 * tf.reduce_sum(tf.square(V)); #original version
        variable_summaries(cost)

    # set optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # init variables
    init = tf.global_variables_initializer()

    # start
    sess.run(init)
    #total_batch = int(sample_num/batch_size);

    # tensorboard
    tf.summary.scalar('loss function', cost)

    summary_dir = './logs/tensorflow/train'
    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)
    tf.gfile.MakeDirs(summary_dir)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
	# Run optimization op (backprop) and cost op (to get loss value)
        _, loss, ypred, resut_prob, true_ind, pi, hinge_loss, sum_loss, f_l_p, np_f_l = sess.run([optimizer, cost, Ypred, res, IND, pi_k, L, PL, f_l_prob, f_l], feed_dict={input_x: Xbase, IND: ind})
        #_, loss, ypred, resut_prob, true_ind, ind_res, pi, hinge_loss, sum_loss, f_l_p, np_f_l = sess.run([optimizer, cost, Ypred, res, IND, mul_ind_res, pi_k, L, PL, f_l_prob, f_l], feed_dict={input_x: Xbase, IND: ind})
        labelSet = np.unique(Ybase)
        Ypred_base = labelSet[ypred]


        if epoch % 10 == 0:
            max_component = np.argmax(pi[0])
            b_l_k = f_l_p[0][:,max_component]

            #print "true", Ybase[0:50]
            #print "pred_idx", ypred[0:50]
            #print "pred", Ypred_base[0:50]

            #print "pi", pi[0,:]
            #print "f_l", np_f_l
            #print "max component index", max_component
            #print "max_label_component", b_l_k
            #print "res", resut_prob[0,:]

            #print "true_ind", true_ind[0, :]
            #print "mul_ind_res", ind_res[0, :]

            #print "L", hinge_loss[0, :]
            print "total_loss", sum_loss

            acc = evaluate_easy(Ypred_base, Ybase)
            print "acc", acc

            abc = sess.run(merged, feed_dict={input_x: Xbase, IND: ind}) # generate summary protobuf object
            writer.add_summary(abc, epoch) # pass the protobuf object to a write, so it can be written on the disk


            # evaluation
            test_BMU, test_A, test_pi = sess.run([BMU, A, pi_k], feed_dict={input_x: Xval})
            test_AS = np.matmul(test_A, Sim_val)
            test_f_l = np.tensordot(test_BMU, test_AS, axes=1)

            test_f_l_prob = sigmoid(test_f_l)
            test_res = np.sum(np.multiply(test_f_l_prob, np.expand_dims(test_pi, 2)), 1)
            test_Ypred = np.argmax(test_res, 1)

            test_labelSet = np.unique(Yval)
            Ypred_val = test_labelSet[test_Ypred]

            test_acc = evaluate_easy(Ypred_val, Yval)
            print "test_acc", test_acc
            #print "test_res", test_res[0,:]
            #print "test_pi", test_pi[0,:]
            #print "test_Ypred", Ypred_val[0:20]
            #print "test_Yture", Yval[0:20]

	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss))
    print '''-----run tensorflow end------'''



    # saving model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print "Model saved in file ", save_path

    return
