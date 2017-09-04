import sys
import sync
import sync_fast
import sync_memory

# load method
def run(data, method, config):
    print '------build model start-----'
    print '------using method: ', method,'---'
    # use sync
    if (method == 'sync'):
        sync.train(data, config)
    elif (method == 'sync_fast'):
        sync_fast.train(data, config)
    elif (method == 'sync_memory'):
        sync_memory.train(data, config)
    print '------build model end------'
