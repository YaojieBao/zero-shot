import sys
import numpy as np
import scipy.spatial.distance as ds

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

''' evaluate '''
def run(Sig_Y, Xval, Yval, Xbase, Ybase, W, sim_scale):
    #Ybase = np.array([1,2,0,3,2]) # test ybase
    #Yval = np.array([1,3])
    Sim_base = Compute_Sim(Sig_Y, Ybase, Ybase, sim_scale)
    Sim_val = Compute_Sim(Sig_Y, Yval, Ybase, sim_scale)
    V = np.matmul(np.linalg.pinv(Sim_base), W)
    Ypred_val = test_V(V, Sim_val, Xval, Yval)
    acc_val = evaluate_easy(Ypred_val, Yval)
    return acc_val
