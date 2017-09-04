import sys
import numpy as np
import sync_memory_learn
import sync_memory_evaluate
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
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim

def train(data, config):
    V_record = [] # training result of W in different folds
    acc_record = [] # validataion accuracy
    nr_fold = config.getint('data', 'nfold')
    lamda = config.getfloat('model', 'lamda')
    sim_scale = config.getfloat('model', 'sim_scale')
    Xtr = data['Xtr']
    Ytr = data['Ytr']
    fold_loc = data['fold_loc']
    Sig_Y = data['Sig_Y']

    # test fold
    # fold_loc = [[0,1],[1,4],[2,4],[1,3],[0,5]]

    #print "before: Xtr", Xtr.shape
    for j in range(nr_fold):
        print "-----------fold: ", j, " ---------"

        #print "delete loc ", j, len(fold_loc[j])
        Xbase = np.delete(Xtr, fold_loc[j], axis=0)
        Ybase = np.delete(Ytr, fold_loc[j], axis=0)
 	Xval = Xtr[fold_loc[j]];
        Yval = Ytr[fold_loc[j]];

        #Sim_base = Compute_Sim(Sig_Y, Ybase, Ybase, sim_scale);

        #print "xbase:", Xbase.shape, " xtr:", Xtr.shape, " Ybase:", Ybase.shape, " Ytr:", Ytr.shape
        V, A = sync_memory_learn.learning(Sig_Y, Xbase, Ybase, lamda)
        acc = sync_memory_evaluate.run(Sig_Y, Xval, Yval, Xbase, Ybase, A, V, sim_scale)
        V_record.append(V)
        acc_record.append(acc)
    print acc_record

    # average accuracy
    acc = 0.0
    acc2 = 0.0
    for accs in acc_record:
        acc = acc + accs[0]
        acc2 = acc2 + accs[1]
    print acc / nr_fold, acc2 / nr_fold

