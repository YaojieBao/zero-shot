import sys
import numpy as np
import random
import tensorflow as tf

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

def learning(Sig_Y, Xbase, Ybase, lamda, Sim_base):
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
    sess = tf.Session()

    # get sizes
    #Ybase = np.array([1,2,0,3,2]) # test ybase
    sample_num, image_size = Xbase.shape # 19443 * 1024
    labels = np.unique(Ybase)
    class_num = labels.shape[0] # 32
    memory_size = Sig_Y.shape[1]
    memory_embedding_size = 100

    # get label index
    ind = -np.ones((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[Ybase == labels[i], i] = 1;

    # input image
    with tf.name_scope('origin_input'):
        input_x = tf.placeholder(tf.float32, shape=[sample_num, image_size]) # 19443, 1024
    with tf.name_scope('dropout_input'):
        X = tf.nn.dropout(input_x, x_dropout_rate)
    with tf.name_scope('U'):
        U = tf.Variable(tf.random_normal([image_size, memory_embedding_size])) # 1024, 100
    with tf.name_scope('u'):
        u = tf.matmul(X, U) # 19443 * 100
        variable_summaries(u)

    # memory lower half ===> generate pi_k, k = 32, memory class size
    with tf.name_scope('origin_memory'):
        input_memory = Sig_Y[np.unique(Ybase-1)] # label embedding 32, 85
    print np.unique(Ybase)
    with tf.name_scope('dropout_memory'):
        memory = tf.nn.dropout(input_memory, m_dropout_rate)
    with tf.name_scope('V'):
        V = tf.Variable(tf.random_normal([memory_size, memory_embedding_size])) # 85, 100
    with tf.name_scope('m'):
        m = tf.matmul(memory, V) # 32, 100
    with tf.name_scope('dotted'):
        dotted = tf.matmul(u, tf.transpose(m)) # 19443, 32
        variable_summaries(dotted)
    with tf.name_scope('pi_k'):
        pi_k = tf.nn.softmax(dotted) # 19443, 32
        variable_summaries(pi_k)

    # memory upper half ===>
    with tf.name_scope('B'):
        B = tf.Variable(tf.random_normal([memory_size, memory_embedding_size])) # 85, 100
        variable_summaries(B)
    with tf.name_scope('BM_0'):
        BM_0 = tf.matmul(memory, B) # [32, 100]
        variable_summaries(BM_0)
    with tf.name_scope('BM_1'):
        BM_1 = tf.expand_dims(BM_0, 0) # [1, 32, 100]
    with tf.name_scope('BM'):
        BM = tf.tile(BM_1, [sample_num, 1, 1]) # [19443, 32, 100]
        variable_summaries(BM)

    # trans each line(100, 1) in u into diag(100, 100), then multiply with BM(32, 100)
    ''' tf.eye(memory_embedding_size) : [100, 100]
        tf.expand_dims([100, 100], 0) : [1, 100, 100]
        tf.tile([1, 100, 100], [19443, 1, 1]) : [19443, 100, 100]
        tf.expand_dims(u, 2) : [19443, 100, 1]
    '''
    with tf.name_scope('huge_diag'):
        huge_diag = tf.tile(tf.expand_dims(tf.eye(memory_embedding_size), 0), [sample_num, 1, 1]) # [19443, 100, 100]
        variable_summaries(huge_diag)
    with tf.name_scope('diag_u'):
        diag_u = tf.multiply(huge_diag, tf.expand_dims(u, 2)) # [19443, 100, 100]
        variable_summaries(diag_u)
    with tf.name_scope('BMU'):
        BMU = tf.matmul(BM, diag_u) # [19443, 32, 100]
        variable_summaries(BMU)

    # expand dimensions for A and Sim_base
    with tf.name_scope('A_0'):
        A_0 = tf.Variable(tf.random_normal([memory_embedding_size, class_num])) # [100, 32], mem
        variable_summaries(A_0)
    with tf.name_scope('A_1'):
        A_1 = tf.expand_dims(A_0, 0) # [1, 100, 32]
    with tf.name_scope('A'):
        A = tf.tile(A_1, [sample_num, 1, 1]) # [19443, 100, 32]
        variable_summaries(A)
    with tf.name_scope('Sim_base'):
        Sim_base = np.tile(np.expand_dims(Sim_base, axis=0), [sample_num, 1, 1])
        variable_summaries(Sim_base)

    with tf.name_scope('BMUA'):
        BMUA = tf.matmul(BMU, A) # [19443, 32, 32]
        variable_summaries(BMUA)
    with tf.name_scope('f_l'):
        f_l = tf.matmul(BMUA, Sim_base)
        variable_summaries(f_l)

    with tf.name_scope('f_l_prob'):
        f_l_prob = tf.transpose(tf.sigmoid(f_l), [0, 2, 1]) # [19443, 32, 32]
        variable_summaries(f_l_prob)
    with tf.name_scope('res'):
        res = tf.reduce_sum(tf.multiply(f_l_prob, tf.expand_dims(pi_k, 2)), 1) # [19443, 32]
        variable_summaries(res)
    with tf.name_scope('Ypred'):
        Ypred = tf.argmax(res, 1)
        variable_summaries(Ypred)

    # loss function
    with tf.name_scope('IND'):
        IND = tf.placeholder(tf.float32, shape=[None, class_num])
        variable_summaries(IND)
    with tf.name_scope('L'):
        mul_ind_res = tf.multiply(IND, res)
        L = tf.square(tf.maximum(0.0, 1 - mul_ind_res))
        PL = tf.reduce_mean(L)
        #PL = tf.nn.l2_loss(L)
        variable_summaries(L)
    #cost = tf.reduce_sum(L) / sample_num / class_num; #original version
    with tf.name_scope('cost'):
        cost = PL + lamda * tf.nn.l2_loss(A) + lamda * tf.nn.l2_loss(B) + lamda * tf.nn.l2_loss(U) + lamda * tf.nn.l2_loss(V)
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
        _, loss, ypred, resut_prob, true_ind, ind_res, pi, hinge_loss, sum_loss, f_l_p, np_f_l = sess.run([optimizer, cost, Ypred, res, IND, mul_ind_res, pi_k, L, PL, f_l_prob, f_l], feed_dict={X: Xbase, IND: ind})
        labelSet = np.unique(Ybase)
        Ypred_base = labelSet[ypred]


        if epoch % 1 == 0:
            max_component = np.argmax(pi[0])
            b_l_k = f_l_p[0][:,max_component]

            print "true", Ybase[0:50]
            print "pred_idx", ypred[0:50]
            print "pred", Ypred_base[0:50]

            print "pi", pi[0,:]
            print "f_l", np_f_l
            print "max component index", max_component
            print "max_label_component", b_l_k
            print "res", resut_prob[0,:]

            print "true_ind", true_ind[0, :]
            print "mul_ind_res", ind_res[0, :]

            print "L", hinge_loss[0, :]
            print "total_loss", sum_loss

            acc = evaluate_easy(Ypred_base, Ybase)
            print "acc", acc

            abc = sess.run(merged, feed_dict={X: Xbase, IND: ind}) # generate summary protobuf object
            writer.add_summary(abc, epoch) # pass the protobuf object to a write, so it can be written on the disk

	# Display logs per epoch step
	if epoch % display_step == 0:
	    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss))
    print '''-----run tensorflow end------'''

    # saving model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print "Model saved in file ", save_path

    rU = sess.run(U)
    rm = sess.run(m)
    rBM = sess.run(BM_1)
    rA = sess.run(A_1)
    return rU, rm, rBM, rA
