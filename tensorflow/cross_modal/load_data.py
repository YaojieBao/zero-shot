import sys
import numpy as np

# one hot embedding
# example: 3 ==> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
def one_hot(y):
    embedding = []
    for i in y:
        tmp = [0] * 10
        tmp[i-1] = 1
        embedding.append(tmp)
    return np.array(embedding)

def load():
    train_x = []
    for line in open('../data/cross_modal/train_x.txt'):
        arr = line.strip().split('  ')
        tmp = [float(i) for i in arr]
        train_x.append(tmp)

    train_y = []
    for line in open('../data/cross_modal/train_y.txt'):
        arr = line.strip().split('  ')
        train_y.append(int(float(arr[0])))

    test_x = []
    for line in open('../data/cross_modal/test_x.txt'):
        arr = line.strip().split('  ')
        tmp = [float(i) for i in arr]
        test_x.append(tmp)

    test_y = []
    for line in open('../data/cross_modal/test_y.txt'):
        arr = line.strip().split('  ')
        test_y.append(int(float(arr[0])))

    return np.array(train_x), one_hot(train_y), np.array(test_x), one_hot(test_y)

