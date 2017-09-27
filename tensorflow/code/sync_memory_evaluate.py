import sys
import numpy as np
import tensorflow as tf
import scipy.spatial.distance as ds
from scipy.special import expit as sigmoid

''' replace nan and inf to 0 '''
def replace_nan(X):
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X

''' compute class label similarity '''
def Compute_Sim(Sig_Y, idx1, idx2, sim_scale):
    sig_y1 = Sig_Y[np.unique(idx1-1)]
    sig_y2 = Sig_Y[np.unique(idx2-1)]

    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim

def construct_W(V, Sim):
    W = np.matmul(Sim, V)
    return W

''' get predict label '''
def test_V(V, Sim, X, Y):
    labelSet = np.unique(Y)
    W = construct_W(V, Sim)
    XW = np.matmul(X, np.transpose(W))
    Ypred = np.argmax(XW, axis=1)
    Ypred = labelSet[Ypred];
    return Ypred

''' get accuracy '''
def evaluate_easy(Ypred, Ytrue):
    labels = np.unique(Ytrue)
    L = labels.shape[0]
    confusion = np.zeros((L, 1))
    for i in range(L):
        confusion[i] = float(np.sum(np.logical_and(Ytrue == labels[i], Ypred == labels[i]))) / np.sum(Ytrue == labels[i])
    acc = np.mean(confusion)
    acc2 = np.mean(Ypred == Ytrue)
    print acc, acc2
    return [acc, acc2]

def softmax(X):
    ''' compute softmax values for each sets of numbers in X. '''
    ''' normalize X first to avoid overflow '''
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    max_val = np.max(X)
    s = np.sum(np.exp(X - max_val), axis = 1)
    return np.exp(X - max_val) / s[:, None]

def compute_probs(A, Sig_Y, Ybase, Yval):
    memory = Sig_Y[np.unique(Ybase - 1)]
    validation = Sig_Y[np.unique(Yval - 1)]
    m = np.matmul(memory, A) # memory_size, memory_embedding_size
    u = np.matmul(validation, A) #input_class_size, memory_embedding_size
    dotted = np.matmul(u, np.transpose(m))
    Sim_val = softmax(dotted)
    return Sim_val

def compute_prob(A, B, Sig_Y, Ybase, Xval):
    memory = Sig_Y[np.unique(Ybase-1)]
    m = np.matmul(memory, A)
    u = np.matmul(Xval, B)
    dotted = np.matmul(u, np.transpose(m))
    Sim_val = softmax(dotted)
    return Sim_val

''' get predict label '''
def test(Sim, X, Y):
    labelSet = np.unique(Y)
    Ypred = np.argmax(Sim, axis=1)
    Ypred = labelSet[Ypred]
    return Ypred

''' evaluate '''
def run(Sig_Y, Xval, Yval, Xbase, Ybase, sim_scale, U, m, BM_1, A_1):
    #Ybase = np.array([1,2,0,3,2]) # test ybase
    #Yval = np.array([1,3])
    #Sim_base = Compute_Sim(Sig_Y, Ybase, Ybase, sim_scale)
    #Sim_val = Compute_Sim(Sig_Y, Yval, Ybase, sim_scale)
    #V = np.matmul(np.linalg.pinv(Sim_base), W)

    #Sim_val = compute_prob(A, B, Sig_Y, Ybase, Xval)
    #Ypred_val = test(Sim_val, Xval, Yval)

    Sim_val = Compute_Sim(Sig_Y, Yval, Ybase, sim_scale)
    sample_num = Xval.shape[0]
    memory_embedding_size = 100

    u = np.matmul(Xval, U)
    dot = np.matmul(u, np.transpose(m))
    pi = softmax(dot)

    BM = np.tile(BM_1, [sample_num, 1, 1])
    huge_diag = np.tile(np.expand_dims(np.eye(memory_embedding_size), 0), [sample_num, 1,1])
    diag_u = np.multiply(huge_diag, np.expand_dims(u, 2))
    BMU = np.matmul(BM, diag_u)

    A = np.tile(A_1, [sample_num, 1, 1])
    BMUA = np.matmul(BMU, A)
    f_l = np.matmul(BMUA, np.transpose(Sim_val))
    f_l_prob = np.transpose(sigmoid(f_l), [0, 2, 1])
    res = np.sum(np.matmul(f_l_prob, np.expand_dims(pi, 2)), 2)

    labelSet = np.unique(Yval)
    Ypred = np.argmax(res, axis=1)
    Ypred_val = labelSet[Ypred]

    acc_val = evaluate_easy(Ypred_val, Yval)
    return acc_val
