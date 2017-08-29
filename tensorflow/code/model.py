import sys
import sync

# load method
def run(data, method, config):
    # use sync
    if (method == 'sync'):
        sync.train(data, config)
