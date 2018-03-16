from optparse import OptionParser
from network import *
import pickle
from net_properties import *
from vocab import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--vocabs", dest="vocabs_path", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--we", type="int", dest="we", default=64)
    parser.add_option("--pe", type="int", dest="pe", default=32)
    parser.add_option("--le", type="int", dest="le", default=32)
    parser.add_option("--hidden1", type="int", dest="hidden1", default=200)
    parser.add_option("--hidden2", type="int", dest="hidden2", default=200)
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000)
    parser.add_option("--epochs", type="int", dest="epochs", default=7) # default=7

    (options, args) = parser.parse_args()

    data_paths = {}
    data_paths['word'] = './data/vocabs.word'
    data_paths['pos'] = './data/vocabs.pos'
    data_paths['label'] = './data/vocabs.labels'
    data_paths['action'] = './data/vocabs.actions'
    data_paths['train'] = './data/train.data'

    if options.model_path and options.vocabs_path:
        net_properties = NetProperties(options.we, options.pe, options.le, options.hidden1, options.hidden2, options.minibatch)

        # creating vocabulary file
        vocab = Vocab(data_paths)

        # writing properties and vocabulary file into pickle
        pickle.dump((vocab, net_properties), open(options.vocabs_path, 'w'))

        # constructing network
        network = Network(vocab, net_properties)

        # training
        network.train(data_paths['train'], options.epochs)

        # saving network
        network.save(options.model_path)


