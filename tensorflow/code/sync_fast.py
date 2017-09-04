import sys
import numpy as np
import sync_fast_learn
import sync_fast_evaluate

def train(data, config):
    W_record = [] # training result of W in different folds
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
        #print "xbase:", Xbase.shape, " xtr:", Xtr.shape, " Ybase:", Ybase.shape, " Ytr:", Ytr.shape
        W = sync_fast_learn.learning(Xbase, Ybase, lamda)
        acc = sync_fast_evaluate.run(Sig_Y, Xval, Yval, Xbase, Ybase, W, sim_scale)
        W_record.append(W)
        acc_record.append(acc)
    print acc_record

    # average accuracy
    acc = 0.0
    acc2 = 0.0
    for accs in acc_record:
        acc = acc + accs[0]
        acc2 = acc2 + accs[1]
    print acc / nr_fold, acc2 / nr_fold

