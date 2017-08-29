import sys
import numpy as np
import ConfigParser
import scipy.spatial.distance as ds

np.seterr(divide='ignore', invalid='ignore')

# one hot embedding
# example: 3 ==> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
def one_hot(y):
    embedding = []
    for i in y:
        tmp = [0] * 10
        tmp[i-1] = 1
        embedding.append(tmp)
    return np.array(embedding)

''' replace nan and inf to 0 '''
def replace_nan(X):
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X

''' normalize label embedding '''
def get_class_signatures(label, norm_method):
    if (norm_method == 'L2'):
        s = np.sqrt(np.sum(np.square(label), axis=1))
        Sig_Y = replace_nan(label / s[:,None])

    Dist = ds.cdist(Sig_Y, Sig_Y, 'euclidean')
    median_Dist = np.median(Dist[Dist>0])
    Sig_Y = replace_nan(Sig_Y / median_Dist)
    return Sig_Y

''' calculate label distance '''
def Sig_dist_comp(Sig_Y):
    inner_product = np.dot(Sig_Y, Sig_Y.transpose())
    C = Sig_Y.shape[0]
    Sig_dist = np.diag(inner_product) * np.ones((1, C)) + np.ones((C, 1)) * np.diag(inner_product).transpose() - 2 * inner_product
    Sig_dist[Sig_dist < 0] = 0
    Sig_dist = np.sqrt(Sig_dist);
    return Sig_dist

''' load AWA dataset '''
def load_AWA(config):
    data_dir = config.get('data', 'AWA_DIR')

    # image feature data: 30475 * 1024 dimension
    X = []
    #for line in open(data_dir+'AWA_googlenet.txt.test'):
    for line in open(data_dir+'AWA_googlenet.txt'):
        arr = line.strip().split('   ')
        tmp = [float(x) for x in arr]
        X.append(tmp)
    X = replace_nan(np.array(X))
    s = np.sqrt(np.sum(np.square(X), axis=1))
    X = replace_nan(X / s[:,None])

    # train image feature: 24295 * 1024 dimension
    tr_idx = [int(line.strip())-1 for line in open(data_dir+'AWA_train_loc.txt')]
    #tr_idx = [int(line.strip())-1 for line in open(data_dir+'AWA_train_loc.txt.test')]
    Xtr = X[tr_idx]

    # test image feature: 6018 * 1024 dimension
    te_idx = [int(line.strip())-1 for line in open(data_dir+'AWA_test_loc.txt')]
    #te_idx = [int(line.strip())-1 for line in open(data_dir+'AWA_test_loc.txt.test')]
    Xte = X[te_idx]

    # class id of training data: 24295 (40 uniq classes)
    Ytr = np.array([int(line.strip()) for line in open(data_dir+'AWA_y_train.txt')])

    # class id of testing data: 6018 (10 uniq classes)
    Yte = np.array([int(line.strip()) for line in open(data_dir+'AWA_y_test.txt')])

    # class order
    class_order = np.array([int(line.strip()) for line in open(data_dir+'class_order')])

    # label embedding: 50 * 85 dimension
    label_embedding = []
    #for line in open(data_dir+'AWA_label_embedding.txt.test'):
    for line in open(data_dir+'AWA_label_embedding.txt'):
        arr = line.strip().split('   ')
        tmp = [float(x) for x in arr]
        label_embedding.append(tmp)
    label_embedding = np.array(label_embedding)
    label_embedding[label_embedding == -1] = 0

    # pack data
    data = {}
    data['Xtr'] = Xtr
    data['Xte'] = Xte
    data['Ytr'] = Ytr
    data['Yte'] = Yte
    data['class_order'] = class_order
    data['label_embedding'] = label_embedding

    return data

''' split data into folds for cross validation '''
def split_fold(data, config):
    nr_fold = config.getint('data', 'nfold')
    Ytr = data['Ytr']
    class_order = data['class_order']
    labelSet = np.unique(Ytr)
    labelSetSize = labelSet.shape[0]
    fold_size = labelSetSize / nr_fold
    fold_loc = []

    for i in range(nr_fold):
        fold_i = []
        for j in range(fold_size):
            cid = labelSet[class_order[i * fold_size + j] - 1]
            fold_i.extend(np.where(Ytr == cid)[0].tolist())
        fold_loc.append(sorted(fold_i));
    return fold_loc

''' load dataset '''
def run(dataset, method, config):
    print "------loading data start------"
    if (dataset == 'AWA'):
	data = load_AWA(config)

    if (method == 'sync'):
        # compute distance for Scr
        Sig_Y = get_class_signatures(data['label_embedding'], config.get('data','norm_method'));
        Sig_dist = Sig_dist_comp(Sig_Y);
        data['Sig_Y'] = Sig_Y
        data['sig_dist'] = Sig_dist

    # split data for cross validation
    fold_loc = split_fold(data, config)
    data['fold_loc'] = fold_loc

    print "------loading data end--------"
    return data
