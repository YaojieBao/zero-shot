import sys
import ConfigParser
import load_data
import model
import evaluate

'''
This is the main function of zero-shot learning methods.
We provide 1 dataset(AWA) and 1 method(Sync)
'''
def main():
    # read config
    config = ConfigParser.ConfigParser()
    config.read('zero-shot.cfg')
    dataset = config.get('data', 'dataset')
    method = config.get('model', 'method')

    # load data
    data = load_data.run(dataset, method, config)

    # train model
    model.run(data, method, config)

if __name__ == "__main__":
    main()
